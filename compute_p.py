import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main")
sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/src")

import yaml
from time import time
import numpy as np
import torch
import hydra
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from pathlib import Path
from tqdm import tqdm

from src.nethook import Trace, set_requires_grad, get_parameter
from src.AlphaEdit_hparams import AlphaEditHyperParams
from src.data_module import TextDatasetQA, custom_data_collator_with_indices, TextDatasetWiki
from src.runningstats import CombinedStat, Mean, NormMean, SecondMoment

model_dict = {
    "phi": "microsoft/phi-1_5",
    "llama2-7b": "NousResearch/Llama-2-7b-chat-hf",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta"
}
STATS_DIR = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/stats"
ROOT_DIR = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main"


def flatten_masked_batch(data, mask):
    """
    Flattens feature data, ignoring items that are masked out of attention.
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]


def layer_stats(
    model,
    layer_name,
    stats_dir,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=None,
    retain_ds=None,
):
    """
    Function to load or compute cached stats.
    """

    # Continue with computation of statistics
    batch_size = 32  # Examine this many dataset texts at once
    if hasattr(model.config, "n_positions"):
        npos = model.config.n_positions
    elif hasattr(model.config, "max_sequence_length"):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, "seq_length"):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config.model_type
        # model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)

    print(f"Computing Cov locally....")
    ds = retain_ds
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: SecondMoment() for k in to_collect})  # {mom2: SecondMoment()}

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=custom_data_collator_with_indices
    )

    # batch_count = -(-(sample_size or len(ds)) // batch_size)  # 向上取整计算批次数量。
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids, labels, attention_mask, indices = batch
            input_ids, labels, attention_mask = input_ids.to("cuda"), labels.to("cuda"), attention_mask.to("cuda")
            with Trace(model, layer_name, retain_input=True, retain_output=False, stop=True) as tr:
                model(input_ids, attention_mask=attention_mask, labels=labels)
            feats = flatten_masked_batch(tr.input, attention_mask)
            # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
            feats = feats.to(dtype=dtype)
            stat.add(feats)
    return stat


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    retain_ds: Dataset = None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    # model_name = model.config._name_or_path.replace("/", "_")
    # key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {layer_name}.")
    stat = layer_stats(
        model,
        layer_name,
        STATS_DIR,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        retain_ds=retain_ds,
    )
    res = stat.mom2.moment().float().to("cpu")  # [in_dim, in_dim]

    return res.to("cuda")


def get_project(model, tok, layer, hparams, retain_ds):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        retain_ds=retain_ds,
    ).cpu()  # K_0 @ K_0^T
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)  # 奇异值分解 得到 U @ S @ U^T
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]  # 找出小于阈值的奇异值的索引
    print(len(small_singular_indices))  # 打印小于阈值的奇异值的数量
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T, cov  # 从U中移除非零奇异值对应的列，得到P


def compute_p(
    model_name="llama2-7b",  # tokenizer path
    # model_path=f"{ROOT_DIR}/tofu_result/locuslab_tofu_ft_phi/models--locuslab--tofu_ft_phi-1.5/snapshots/c34da369771ff53cf722344bc912fd9913f67da7",  # pretrained model path or id
    model_path=f"locuslab/tofu_ft_llama2-7b",  # pretrained model path or id
	split="retain99",
    data_name="tofu",
    data_path="locuslab/TOFU",
):
    # =============================== load model ===============================
    model_id = model_dict[model_name]
    if model_path is None:
        model_path = model_id

    print("Instantiating model")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        use_flash_attention_2=False,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    # model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # =============================== load hparams ===============================
    params_path = f"{ROOT_DIR}/config/edit/{model_name}.json"
    hparams = AlphaEditHyperParams.from_json(params_path)
    print(f"Executing with parameters {hparams}")

    # =============================== load retain data ===============================
    max_length = 512  # tokenizer.model_max_length
    if data_name == "tofu":
        retain_ds = TextDatasetQA(
            data_path,
            tokenizer=tokenizer,
            model_family=model_name,
            max_length=max_length,
            split=split,
            question_key="question",
            answer_key="answer",
        )  # [input_ids, label, attention_mask, indices]
    elif data_name == "wmdp":
        retain_ds = TextDatasetWiki(
            data_path,
            tokenizer=tokenizer,
            model_family=model_name,
            max_length=max_length,
            split=split,
        )
    # ===============================  ===============================
    # Iterate through dataset
    W_out = get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
    )  # hparams.layers[-1]的权重 shape [in_dim, out_dim] [2048, 8192]
    P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
    cache_K = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
    del W_out
    layers_name = "".join(str(i) for i in hparams.layers)
    # 计算每层的投影矩阵
    proj_path = f"{STATS_DIR}/{model_name}/{data_name}_{split}_{layers_name}_null_space_project.pt"
    # cache_path = f"{STATS_DIR}/{model_name}/{data_name}_{split}_{layers_name}_cache_k0.pt"
    if os.path.exists(proj_path):
        return
    else:
        for i, layer in enumerate(hparams.layers):
            P[i, :, :], cache_K[i, :, :] = get_project(model, tokenizer, layer, hparams, retain_ds)
        
        proj_path_obj = Path(proj_path)
        proj_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(P, proj_path)
        print(f"projection matrix saved to {proj_path}")
        # torch.save(cache_K, cache_path)

def read_yaml_file(file_path):
    """
    读取 YAML 文件并将其转换为字典格式。

    :param file_path: YAML 文件的路径
    :return: 包含 YAML 数据的字典
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            # 使用 yaml.safe_load 安全加载 YAML 文件内容
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"读取 YAML 文件时出错: {e}")
            return None


if __name__ == "__main__":
    config_path = f"{ROOT_DIR}/config/forget/forget_zephyr_wmdp.yaml"
    args = read_yaml_file(config_path)

    print(args["model_family"], args["model_path"], args["split"])
    compute_p(args["model_family"], args["model_path"], args["split"], args["data_name"], args["data_path"])
