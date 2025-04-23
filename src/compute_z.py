from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import nethook, repr_tools
from src.AlphaEdit_hparams import AlphaEditHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    # target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]
    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]

    # Compile list of rewriting and KL x/y pairs
    # rewriting_prompts, kl_prompts = [
    #     context.format(request["prompt"]) + tok.decode(target_ids[:-1])
    #     for context_types in context_templates
    #     for context in context_types
    # ], ["{} is a"]
    # all_prompts = rewriting_prompts + kl_prompts

    # input_tok = tok(
    #     [prompt.format(request["subject"]) for prompt in all_prompts],
    #     return_tensors="pt",
    #     padding=True,
    # ).to("cuda")

    # 将目标标记 target_new 填充到相应位置上 Compute rewriting targets,
    # rewriting_targets = torch.tensor(-100, device="cuda").repeat(
    #     len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    # )
    # for i in range(len(rewriting_prompts)):
    #     ex_len = input_tok["attention_mask"][i].sum()
    #     rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # 计算每个提示中事实的索引 Compute indices of the tokens where the fact is looked up
    # lookup_idxs = [
    #     find_fact_lookup_idx(prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0))
    #     for i, prompt in enumerate(all_prompts)
    # ]

    # 选择损失计算的层，取超参数指定的损失层和重写层的最大值 Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    max_length = len(request["labels"])
    if hasattr(model.config, "n_embd"):
        delta = torch.zeros(
            (
                max_length,
                model.config.n_embd,
            ),
            requires_grad=True,
            device="cuda",
        )
    elif hasattr(model.config, "hidden_size"):
        delta = torch.zeros(
            (
                max_length,
                model.config.hidden_size,
            ),
            requires_grad=True,
            device="cuda",
        )
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # 在指定层的输出中插入 delta 变量，并记录初始值
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, :].detach().clone()  # [max_length, hidden_dim,]
                # cur_out[0].shape = torch.Size([1,max_length, hidden_dim])

            # Add intervened delta
            # for i, idx in enumerate(lookup_idxs):
            #     if len(lookup_idxs) != len(cur_out[0]):
            #         cur_out[0][idx, i, :] += delta
            #     else:
            #         cur_out[0][i, idx, :] += delta
            cur_out[0][0, :, :] += delta

        return cur_out

    # Optimizer 使用 Adam 优化器优化 delta，并冻结模型的其他参数
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,  # 在指定层的输出中插入 delta 变量
        ) as tr:
            logits = model(
                request["input_ids"].unsqueeze(0),
                labels=request["labels"].unsqueeze(0),
                attention_mask=request["attention_mask"].unsqueeze(0),
            ).logits  # model(**input_tok).logits

            # # Compute distribution for KL divergence
            # kl_logits = torch.stack(
            #     [logits[i - len(kl_prompts), idx, :] for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])],
            #     dim=0,
            # )  # [1, vocab_size]
            # kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            # if kl_distr_init is None:
            #     kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        rewriting_targets = request["labels"].unsqueeze(0)  # shape [1, max_length]

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]  # [1, max_length, hidden_dim]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output  # [: len(rewriting_prompts)]  # [batch_size, seq_len, hidden_size]

        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2
        )  # [batch_size, seq_len, vocab_size]
        loss = torch.gather(
            log_probs,  # input
            2,  # dim
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(
            2
        )  # [bs, max_len]
        mask = (rewriting_targets != -100).float()  # [bs, max_len]

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / rewriting_targets.size(0)
        nll_loss = nll_loss_each.mean()
        # kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
        #     kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        # )
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + weight_decay.to(nll_loss.device)  # + kl_loss.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "  #  + {np.round(kl_loss.item(), 3)}
            f"avg prob is "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break
        if it == hparams.v_num_grad_steps - 1:
            break
        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
    # FIXME: 根据mask进行filter，支输出label部分
    target_init = flatten_masked_batch(target_init, request["attention_mask"])
    delta = flatten_masked_batch(delta, request["attention_mask"])
    target = target_init + delta
    print(f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}")

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],  # subject
    module_template: str,  # layer_name
    fact_token_strategy: str,  # "subject_last"
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    context_info = dict(
        context_templates=context_templates,
        words=words,
    )
    subtoken = fact_token_strategy[len("subject_") :]  # last
    l_input, l_output = repr_tools.get_reprs_at_word_tokens(
        track="both", subtoken=subtoken, **context_info, **word_repr_args
    )

    return l_input.detach(), l_output.detach()


def get_module_input_output(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    requests: List,  # subject
    module_template: str,  # layer_name
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    def _batch(n):
        for i in range(0, len(requests), n):
            yield requests[i : i + n]  # 将句子和被填词位置分块

    module_name = module_template.format(layer)
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
                flatten_masked_batch(tr.output[0], attention_mask)
            )  # _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.concat(v, 0) for k, v in to_return.items() if len(v) > 0}

    return to_return["in"].detach(), to_return["out"].detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]
