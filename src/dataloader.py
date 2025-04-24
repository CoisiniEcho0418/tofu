import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
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


def printll(name, inp):
    # print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
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
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def custom_data_collator_forget_dpo(samples):
    rets = []
    # no need to add idk, just use the refuse_label as the idk
    res = {}

    # ["idk", "forget", "retain"]
    if samples[0][0]:
        idk_samples = [sample[0] for sample in samples]
        res["idk"] = (
            torch.stack([sample[0] for sample in idk_samples]),
            torch.stack([sample[1] for sample in idk_samples]),
            torch.stack([sample[2] for sample in idk_samples]),
        )
        rets.append(res["idk"])
    if samples[0][1]:
        forget_samples = [sample[1] for sample in samples]
        res["forget"] = (
            torch.stack([sample[0] for sample in forget_samples]),
            torch.stack([sample[1] for sample in forget_samples]),
            torch.stack([sample[2] for sample in forget_samples]),
        )
        rets.append(res["forget"])

    if samples[0][2]:
        retain_samples = [sample[2] for sample in samples]
        res["retain"] = (
            torch.stack([sample[0] for sample in retain_samples]),
            torch.stack([sample[1] for sample in retain_samples]),
            torch.stack([sample[2] for sample in retain_samples]),
        )
        rets.append(res["retain"])

    return rets

    # def compute_metrics(pred):
    #     logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    #     preds = torch.from_numpy(pred.predictions.argmax(-1))
    #     shifted_labels = labels[..., 1:].contiguous()
    #     acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    #     loss = get_loss(logits, labels)
    #     return {"eval accuracy": acc, "eval loss": loss.item()}

    # def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss
