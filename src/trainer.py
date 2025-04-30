import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json
from pathlib import Path
from data_module import get_batch_loss
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers import LlamaForCausalLM, PhiForCausalLM
from peft import PeftModelForCausalLM, LoraModel


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction="none").sum(-1)  # (bs*(seq_len - 1))

    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    input_ids, labels, attention_mask = inputs
    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(input_ids, labels=labels, attention_mask=attention_mask)
    else:
        _ = model(input_ids, labels=labels, attention_mask=attention_mask)

    hook_handle.remove()

    return cache[0]


class ProbLossStable(nn.Module):
    def __init__(self, reduction="none", eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        # self._softmax = nn.LogSoftmax(dim=-1)
        # self._nllloss = nn.NLLLoss(reduction='none')
        self._nllloss = nn.NLLLoss(reduction="none", ignore_index=-100)

    def forward(self, outputs, labels):
        return self._nllloss(self._softmax(outputs), labels)


def get_contrastive_loss(prob_sum_unlearn, prob_sum_good, div="TV"):

    # div = 'KL'
    # div = 'Jenson-Shannon'
    # div = 'Pearson'
    if div == "KL":

        def activation(x):
            return -torch.mean(x)

        def conjugate(x):
            return -torch.mean(torch.exp(x - 1.0))

    elif div == "Reverse-KL":

        def activation(x):
            return -torch.mean(-torch.exp(x))

        def conjugate(x):
            return -torch.mean(-1.0 - x)  # remove log

    elif div == "Jeffrey":

        def activation(x):
            return -torch.mean(x)

        def conjugate(x):
            return -torch.mean(x + torch.mul(x, x) / 4.0 + torch.mul(torch.mul(x, x), x) / 16.0)

    elif div == "Squared-Hellinger":

        def activation(x):
            return -torch.mean(1.0 - torch.exp(x))

        def conjugate(x):
            return -torch.mean((1.0 - torch.exp(x)) / (torch.exp(x)))

    elif div == "Pearson":

        def activation(x):
            return -torch.mean(x)

        def conjugate(x):
            return -torch.mean(torch.mul(x, x) / 4.0 + x)

    elif div == "Neyman":

        def activation(x):
            return -torch.mean(1.0 - torch.exp(x))

        def conjugate(x):
            return -torch.mean(2.0 - 2.0 * torch.sqrt(1.0 - x))

    elif div == "Jenson-Shannon":

        def activation(x):
            return -torch.mean(-torch.log(1.0 + torch.exp(-x))) - torch.log(torch.tensor(2.0))

        def conjugate(x):
            return -torch.mean(x + torch.log(1.0 + torch.exp(-x))) + torch.log(torch.tensor(2.0))

    elif div == "TV":

        def activation(x):
            return -torch.mean(torch.tanh(x) / 2.0)

        def conjugate(x):
            return -torch.mean(torch.tanh(x) / 2.0)

    else:
        raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)

    prob_reg = -prob_sum_good
    loss_regular = activation(prob_reg)
    prob_peer = -prob_sum_unlearn
    loss_peer = conjugate(prob_peer)
    # print("current:", loss_regular, loss_peer)
    loss = loss_regular - loss_peer
    return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("forget_loss")
        self.oracle_model = kwargs.pop("oracle_model")
        self.eval_cfg = kwargs.pop("eval_cfg")
        self.beta = 0.1

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        if self.loss_type == "RMU":
            layer_id = 7
            steering_coeff_list = 20

            self.frozen_module = self._get_model_layer(self.oracle_model, layer_id)
            self.updated_module = self._get_model_layer(self.model, layer_id)

            random_vector = torch.rand(
                1, 1, self.model.config.hidden_size, dtype=self.model.dtype, device=self.model.device
            )
            self.control_vec = random_vector / torch.norm(random_vector) * steering_coeff_list

    def _get_model_layer(self, model, layer_id):
        """
        通用方法：根据模型类型和层ID获取指定的模型层（适用Peft-llama,llama,Phi）
        """
        # 如果模型是 PeftModelForCausalLM 类型
        if isinstance(model, PeftModelForCausalLM):
            base_model = model.base_model  # 访问下一层：Lora 加速模型
            if isinstance(base_model, LoraModel):  
                base_model = base_model.model  # 继续访问到 Llama 模型
            if isinstance(base_model, LlamaForCausalLM):  
                return base_model.model.layers[layer_id]
        if isinstance(model, LlamaForCausalLM) or isinstance(model, PhiForCausalLM):
            return model.model.layers[layer_id]
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss

        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(
                    retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask
                )

            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction="batchmean", log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":  # idk loss + retain loss
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

            # concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)

            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

        elif self.loss_type == "dpo":  # idk as positive, forget as negative
            # idk_inputs, forget_inputs = inputs
            # idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            # forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(
                    idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask
                )
                forget_outputs_oracle = self.oracle_model(
                    forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask
                )
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)

            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

            outputs = forget_outputs

        elif self.loss_type == "npo":
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

        elif self.loss_type == "ME":
            forget_inputs, retain_inputs = inputs
            # get forget loss
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            outputs = model(forget_input_ids, labels=None, attention_mask=forget_attention_mask)
            forget_loss = get_me_loss(outputs.logits, forget_labels)

            # get retain loss
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            rt_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            rt_loss = rt_outputs.loss
            # rt_loss = rt_loss.mean()
            loss = 0.1 * forget_loss + 1.0 * rt_loss

        elif "FLAT" in self.loss_type:
            cl_div = self.loss_type.split("-")[-1]

            idk_inputs, forget_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs

            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
            idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)

            losses_unlearn = []
            losses_good = []

            shift_logits = forget_outputs.logits[:, :-1, :]
            shift_labels_unlearn = forget_labels[:, 1:]
            shift_logits_good = idk_outputs.logits[:, :-1, :]
            shift_labels_good = idk_labels[:, 1:]

            criterion_prob = ProbLossStable()

            for bid in range(forget_input_ids.shape[0]):
                loss_unlearn = criterion_prob(shift_logits[bid], shift_labels_unlearn[bid])
                loss_good = criterion_prob(shift_logits_good[bid], shift_labels_good[bid])
                losses_unlearn.append(loss_unlearn)
                losses_good.append(loss_good)

            loss_sum_unlearn = torch.stack(losses_unlearn).mean()
            loss_sum_good = torch.stack(losses_good).mean()

            loss = get_contrastive_loss(loss_sum_unlearn, loss_sum_good, cl_div)

        elif self.loss_type == "RMU":
            forget_inputs, retain_inputs = inputs  
            input_ids, labels, attention_mask = forget_inputs

            # Unlearning loss
            if isinstance(model, DataParallel):
                updated_forget_activations = forward_with_cache(
                    model.module, forget_inputs, module=self.updated_module, no_grad=False
                )
            else:
                updated_forget_activations = forward_with_cache(
                    model, forget_inputs, module=self.updated_module, no_grad=False
                )  # [bs, max_length, hidden_size]
            unlearn_loss = torch.nn.functional.mse_loss(
                updated_forget_activations, self.control_vec.to(updated_forget_activations.device)
            )

            # Retain loss
            if isinstance(model, DataParallel):
                updated_retain_activations = forward_with_cache(
                    model.module, retain_inputs, module=self.updated_module, no_grad=False
                )
            else:
                updated_retain_activations = forward_with_cache(
                    model, retain_inputs, module=self.updated_module, no_grad=False
                )
            frozen_retain_activations = forward_with_cache(
                self.oracle_model, retain_inputs, module=self.frozen_module, no_grad=True
            )

            retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, frozen_retain_activations)
            retain_loss *= 100  # alpha

            # Update model
            loss = unlearn_loss.to(self.model.device) + retain_loss.to(self.model.device)

        return (loss, outputs) if return_outputs else loss

    # def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
    #     input_ids, labels, attention_mask = inputs
    #     # forward pass
    #     with torch.no_grad():
    #         outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    #         logits = outputs.logits
    #         loss = outputs.loss
    #     return (loss, logits, labels)

    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     ignore_keys=None,
    #     metric_key_prefix="eval",
    # ):
    #     # if eval is called w/o train, handle model prep here
    #     if self.is_deepspeed_enabled and self.deepspeed is None:
    #         _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
    #     args = self.args
    #     model = self._wrap_model(self.model, training=False, dataloader=None)
    #     print(
    #         self.is_in_train,
    #         args.device,
    #         self.model.dtype,
    #         self.args.dataloader_num_workers,
    #         self.eval_cfg.split_list,
    #         self.eval_cfg.split,
    #     )
    #     if len(self.accelerator._models) == 0 and model is self.model:
    #         model = (
    #             self.accelerator.prepare(model)
    #             if self.is_deepspeed_enabled
    #             else self.accelerator.prepare_model(model, evaluation_mode=True)
    #         )

    #         if self.is_fsdp_enabled:
    #             self.model = model

    #         # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #         if model is not self.model:
    #             self.model_wrapped = model

    #         # backward compatibility
    #         if self.is_deepspeed_enabled:
    #             self.deepspeed = self.model_wrapped

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)
    #     model.eval()
    #     curr_step = self.state.global_step
    #     eval_cfg = self.eval_cfg

    #     curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
    #     Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
    #     with torch.no_grad():
    #         for i, (
    #             folder,
    #             split,
    #             question_key,
    #             answer_key,
    #             eval_task,
    #             base_answer_key,
    #             perturbed_answer_key,
    #         ) in enumerate(
    #             zip(
    #                 eval_cfg.data_path,
    #                 eval_cfg.split_list,
    #                 eval_cfg.question_key,
    #                 eval_cfg.answer_key,
    #                 eval_cfg.eval_task,
    #                 eval_cfg.base_answer_key,
    #                 eval_cfg.perturbed_answer_key,
    #             )
    #         ):
    #             world_size = self.accelerator.num_processes

    #             # For some reason, Hydra is not interprating the split correctly
    #             if eval_task == "eval_log_forget":
    #                 split = eval_cfg.split
    #             print(f"Working on eval task {eval_task} with split {split}")
    #             save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
    #             save_filename = (
    #                 save_filename
    #                 if world_size == 1
    #                 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
    #             )
    #             # print(save_filename)
    #             if os.path.exists(save_filename) and not eval_cfg.overwrite:
    #                 print(f"Skipping {eval_task} because {save_filename} already exists")
    #                 continue

    #             eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
    #                 eval_cfg,
    #                 eval_task,
    #                 self.tokenizer,
    #                 folder,
    #                 split,
    #                 question_key,
    #                 answer_key,
    #                 base_answer_key,
    #                 perturbed_answer_key,
    #             )
    #             eval_dataloader = self.accelerator.prepare(eval_dataloader)
    #             # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
    #             base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
    #             perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
    #             normalize_gt = False
    #             # if 'eval_log' not in eval_task:
    #             #     normalize_gt = True

    #             eval_logs = get_all_evals(
    #                 eval_cfg,
    #                 model,
    #                 self.tokenizer,
    #                 eval_task,
    #                 eval_dataloader,
    #                 base_eval_dataloader,
    #                 perturb_dataloader,
    #                 normalize_gt=normalize_gt,
    #             )

    #             with open(save_filename, "w") as f:
    #                 # pretty write json to f
    #                 json.dump(eval_logs, f, indent=4)

    #             # wait for all process to finish
    #         self.accelerator.wait_for_everyone()
    #         aggregated_eval_logs = {}
    #         for eval_task in eval_cfg.eval_task:
    #             # read the saved file as json and merge them using merge_dicts
    #             if world_size > 1:
    #                 if self.accelerator.is_local_main_process:
    #                     eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
    #                     for i in range(1, world_size):
    #                         filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
    #                         eval_logs = merge_dicts(eval_logs, json.load(open(filename)))

    #                     aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

    #                     new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
    #                     with open(new_save_filename, "w") as f:
    #                         # pretty write json to f
    #                         json.dump(eval_logs, f, indent=4)

    #                         # delete old files use shutil

    #                         for i in range(world_size):
    #                             filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
    #                             os.remove(filename)

    #         if self.accelerator.is_local_main_process:
    #             # aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
    #             aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

    #             with open(aggregated_eval_log_filename, "w") as f:
    #                 json.dump(aggregated_eval_logs, f, indent=4)

    #             if eval_cfg.retain_result is not None:
    #                 model_utility = get_model_utility(aggregated_eval_logs)
    #                 retain_result = json.load(open(eval_cfg.retain_result, "r"))
    #                 forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
    #                 aggregate_stat = {**model_utility, **forget_quality}

    #                 # save aggregate_stat as csv
    #                 with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), "w") as csvfile:
    #                     field_names = list(aggregate_stat.keys())
    #                     writer = csv.DictWriter(csvfile, fieldnames=field_names)
    #                     writer.writeheader()
    #                     writer.writerow(aggregate_stat)
