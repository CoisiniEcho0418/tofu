import sys
import os
sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main")
sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/src")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, LlamaForCausalLM, PhiForCausalLM

import hydra
import transformers
from peft import LoraConfig, get_peft_model, PeftModel, PeftModelForCausalLM, LoraModel
from pathlib import Path
from src.utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf

from src.data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA, WmdpForgetDatasetQA
from src.dataloader import custom_data_collator_forget, custom_data_collator_forget_dpo
from src.trainer import CustomTrainerForgetting

dataset_class_mapping = {
    "tofu": TextForgetDatasetQA,
    "wmdp": WmdpForgetDatasetQA,
}


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_params(model, layer_ids, param_ids):
    """
    获取指定模型层的参数。

    Args:
        model: 模型实例，可以是 PeftModelForCausalLM、LlamaForCausalLM 或 PhiForCausalLM。
        layer_ids: 要提取参数的层索引列表。
        param_ids: 要提取参数的参数索引列表。

    Returns:
        params: 提取的参数列表。
    """
    # 验证输入参数
    if not isinstance(layer_ids, list) or not isinstance(param_ids, list):
        raise ValueError("layer_ids 和 param_ids 必须是列表类型")
    params = []

    def _get_layer_params(layers, layer_ids, param_ids):
        """从指定的层中提取参数"""
        for layer_id in layer_ids:
            for i, p in enumerate(layers[layer_id].parameters()):
                if i in param_ids:
                    params.append(p)

    # 如果模型是 PeftModelForCausalLM 类型
    if isinstance(model, PeftModelForCausalLM):
        base_model = model.base_model  # 访问下一层：Lora 加速模型
        if isinstance(base_model, LoraModel):  # Lora 加速模型
            base_model = base_model.model  # 继续访问到 Llama 模型
        if isinstance(base_model, LlamaForCausalLM):  # 如果是 Llama 模型
            _get_layer_params(base_model.model.layers, layer_ids, param_ids)
        else:
            raise ValueError(f"Unsupported model type inside base_model: {type(base_model)}")

    # 如果是 Llama 类型的模型
    elif isinstance(model, LlamaForCausalLM):
        _get_layer_params(model.model.layers, layer_ids, param_ids)

    # 如果是 Phi 类型的模型
    elif isinstance(model, PhiForCausalLM):
        _get_layer_params(model.model.layers, layer_ids, param_ids)

    # 不支持的模型类型
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    return params


@hydra.main(
    version_base=None,
    config_path="/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/config/forget",
    config_name="forget_zephyr_wmdp",
)
def main(cfg):
    # TODO:
    # cfg.lr = 1e-4
    # cfg.weight_decay = 0.01
    # cfg.batch_size = 16
    # cfg.gradient_accumulation_steps = 8
    # cfg.num_epochs = 10
    # cfg.split = "forget01" if cfg.data_name == "tofu" else None  #
    # cfg.forget_loss = "npo"  # ["grad_ascent", "grad_diff", "idk", "npo", "dpo", "ME", "FLAT-TV", "RMU"]
	
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    print(f"num_devices: {num_devices}")

    # if os.environ.get("LOCAL_RANK") is not None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # =============================== load data ===============================
    max_length = 1024  # tokenizer.model_max_length
    if cfg.forget_loss in ["dpo"]:
        torch_format_dataset = TextForgetDatasetDPOQA(
            cfg.data_path, 
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            max_length=max_length, 
            split=cfg.split
        )
    else:
        torch_format_dataset = dataset_class_mapping[cfg.data_name](
            cfg.data_path,
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            max_length=max_length,
            split=cfg.split,
            loss_type=cfg.forget_loss,
        )

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices))

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    print(f"---len(torch_format_dataset): {len(torch_format_dataset)}---")
    print(f"max_steps: {max_steps}")
    

    # first get the base model architectur2e
    # if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re

    # =============================== load model ===============================
    path_found = False
    if cfg.model_path != model_cfg["ft_model_path"]:
        for file in os.listdir(cfg.model_path):
            if re.search(r"pytorch.*\.bin", file):
                path_found = True
                break

            if re.search(r"model-*\.safetensors", file):
                path_found = True
                break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)
        print("Loading from checkpoint", cfg.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=model_cfg["flash_attention2"] == "true",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
        if cfg.forget_loss in ["KL", "dpo", "npo", "RMU"]:
            oracle_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=config,
                use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map,
            )        
    else:
        print("Error! Model not found in the path")
        return
        
    
    # else:
    #     print("Loading after merge and unload")
    #     model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, device_map=device_map)
    #     #now use the checkpoint to add the LoRA modules
    #     model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
    #     #save this as a standard model so that we can again do PEFT style finetuneing from scratch
    #     model = model.merge_and_unload()
    #     #save the model for next time
    #     model.save_pretrained(cfg.model_path)

    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    # =============================== config Trainer ===============================
    # now we have a HuggingFace model
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r,
        lora_alpha=cfg.LoRA.alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=cfg.LoRA.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    if cfg.forget_loss in ["dpo"]:
        data_collator_ = custom_data_collator_forget_dpo
    else:
        data_collator_ = custom_data_collator_forget

    # config optimizer
    if cfg.forget_loss == "RMU":
        layer_ids = [5, 6, 7]
        param_ids = [6]
        params = get_params(model, layer_ids, param_ids)
        optimizer = torch.optim.AdamW(params, lr=cfg.lr)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = None

    # config args
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=1,  # max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=10, #max(1, max_steps // 20),
        logging_dir=f"{cfg.save_dir}/logs",
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=max_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        # deepspeed="/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/config/ds_config.json",
        weight_decay=cfg.weight_decay,
        eval_steps=steps_per_epoch,
        eval_strategy="steps" if cfg.eval_while_train else "no",
        seed=cfg.seed,
    )

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,  # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=data_collator_,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
        optimizers=(optimizer, None),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # =============================== start train() ===============================
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    # save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        print("######################")
        print("Saving to: ", cfg.save_dir)
        print("######################")

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                # delete the directory
                import shutil

                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
