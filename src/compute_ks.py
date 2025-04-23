from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compute_z import get_module_input_output_at_words, flatten_masked_batch
from src.AlphaEdit_hparams import AlphaEditHyperParams
from src import nethook


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
):
    def _batch(n):
        for i in range(0, len(requests), n):
            yield requests[i : i + n]  # 将句子和被填词位置分块

    module_name = hparams.rewrite_module_tmp.format(layer)
    to_return = {"in": [], "out": []}

    for batch_contexts in _batch(n=128):
        with torch.no_grad():
            input_ids = [s["input_ids"] for s in batch_contexts]
            labels = [s["labels"] for s in batch_contexts]
            attention_mask = [s["attention_mask"] for s in batch_contexts]
            batch_idxs = [s["case_id"] for s in batch_contexts]

            input_ids = torch.stack(input_ids).to(model.device)
            labels = torch.stack(labels).to(model.device)
            attention_mask = torch.stack(attention_mask).to(model.device)
            batch_idxs = torch.stack(batch_idxs)
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=True,
                retain_output=True,
            ) as tr:
                model(input_ids, labels=labels, attention_mask=attention_mask)

            to_return["in"].append(
                flatten_masked_batch(tr.input, attention_mask)
            )  # _process(tr.input, batch_idxs, "in")
            to_return["out"].append(
                flatten_masked_batch(tr.output, attention_mask)
            )  # _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.concat(v, 0) for k, v in to_return.items() if len(v) > 0}
    return to_return["in"]


# def compute_ks(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: Dict,
#     hparams: AlphaEditHyperParams,
#     layer: int,
#     context_templates: List[str],
# ):
#     layer_ks = get_module_input_output_at_words(
#         model,
#         tok,
#         layer,
#         context_templates=[
#             context.format(request["prompt"])
#             for request in requests
#             for context_type in context_templates
#             for context in context_type
#         ],
#         words=[request["subject"] for request in requests for context_type in context_templates for _ in context_type],
#         module_template=hparams.rewrite_module_tmp,
#         fact_token_strategy=hparams.fact_token,
#     )[
#         0
#     ]  # 计算特定层的输入表示

#     context_type_lens = [0] + [len(context_type) for context_type in context_templates]
#     context_len = sum(context_type_lens)  # template的数量
#     context_type_csum = np.cumsum(context_type_lens).tolist()

#     ans = []
#     for i in range(0, layer_ks.size(0), context_len):
#         tmp = []
#         for j in range(len(context_type_csum) - 1):
#             start, end = context_type_csum[j], context_type_csum[j + 1]
#             tmp.append(layer_ks[i + start : i + end].mean(0))
#         ans.append(torch.stack(tmp, 0).mean(0))
#     return torch.stack(ans, dim=0)
