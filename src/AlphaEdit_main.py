import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import unicodedata

from src import nethook
from src.globals import *

from src.compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, get_module_input_output
from .AlphaEdit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cfg,
    cache_template: Optional[str] = None,
    cache_c=None,
    cache_k=None,
    P=None,
    use_cache=True,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    Args:
        model (AutoModelForCausalLM): 要更新的模型。
        tok (AutoTokenizer): 用于处理输入的分词器。
        requests (List[Dict]): 包含更新请求的列表，每个请求是一个字典。
        hparams (AlphaEditHyperParams): 超参数对象。
        cache_template (Optional[str], optional): 缓存模板字符串，默认为 None。
        cache_c: 缓存矩阵，默认为 None。
        P: 投影矩阵，默认为 None。

    Returns:
        Dict[str, Tuple[torch.Tensor]]: 更新后的模型和更新后的缓存矩阵。

    """

    # Update target and print info
    requests = deepcopy(requests)
    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Compute z for final layer
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(str(cache_template).format(z_layer, hparams.clamp_norm_factor, request["case_id"]))
            if cache_template is not None
            else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():  # Require cache template  # Cache file must exist
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                None,  # context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.concat(z_list, dim=0)  # [total_seq_length, out_dim]
    # =============================== 遍历每个层进行更新 ===============================
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, None).T  # K1 [in_dim, total_seq_length]
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output(
            model,
            tok,
            z_layer,
            context_templates=None,
            requests=requests,
            module_template=hparams.layer_module_tmp,
        )[
            1
        ]  # W * K1  z_layer output [total_seq_length, out_dim]
        if cfg.forget_loss == "edit_u": # ga
            zs = -zs
        targets = (zs - cur_zs).T  # R = V1 - W * K1 [out_dim, total_seq_length, ]
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(
            repeat_factor, dim=1
        )  # 沿着第二维（dim=1）对 targets 张量进行重复扩展，使 targets 张量的第二维大小与 layer_ks 张量的第二维大小相匹配。
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        if cfg.forget_loss in ["edit_t", "edit_u"]:  # use null space projection
            upd_matrix = torch.linalg.solve(
                P[i, :, :].cuda() @ (layer_ks @ layer_ks.T + cache_c[i, :, :].cuda())
                + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
                P[i, :, :].cuda() @ layer_ks.to(P.dtype) @ resid.T,
            )  # 求解 Ax = b
            # A = P @ (K1 @ K1.T + KP @ KP.T) + I
            # b = P @ K1 @ R.T
        elif cfg.forget_loss == "edit_k0":  # use cached k0
            if cache_k is None:
                print("Cache k not found for edit_k0 loss!")
                return None, None
            upd_matrix = torch.linalg.solve(
                (layer_ks @ layer_ks.T + hparams.mom2_update_weight * cache_k[i, :, :].cuda()),
                layer_ks.to(P.dtype),
            )
            upd_matrix = resid @ upd_matrix.T
        elif cfg.forget_loss == "edit_max":
            # - forget loss =  -||WK-V|| + ||\Delta P||
            upd_matrix = torch.linalg.solve(
                P[i, :, :].cuda() @ (-layer_ks @ layer_ks.T + cache_c[i, :, :].cuda())
                + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
                -P[i, :, :].cuda() @ layer_ks.to(P.dtype) @ resid.T,
            )
            # A = P @ (- K1 @ K1.T + KP @ KP.T) + I
            # b = - P @ K1 @ R.T
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # Clear GPU memory
        # del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    if use_cache:
        for i, layer in enumerate(hparams.layers):
            print(f"Saving cache for layer: {layer} ({i})")
            layer_ks = compute_ks(model, tok, requests, hparams, layer, None).T  # [in_dim, total_seq_length]
            # 随机采样部分列
            sample_size = 500
            total_seq_length = layer_ks.size(1)
            sample_indices = torch.randperm(total_seq_length)[:min(sample_size, total_seq_length)]
            sampled_layer_ks = layer_ks[:, sample_indices]
            cache_c[i, :, :] += sampled_layer_ks.cpu() @ sampled_layer_ks.cpu().T
            # cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. " "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(next(model.parameters()).device)
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1)
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [unicodedata.normalize("NFKD", x).replace("\n\n", " ").replace("<|endoftext|>", "") for x in txt]

    return txt
