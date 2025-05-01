import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main")
sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/src")

from omegaconf import OmegaConf
import hydra
import json
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

from evaluate_util import get_all_evals, get_dataloader
from src.utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality


@hydra.main(
    version_base=None,
    config_path="/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/config/eval",
    config_name="eval_everything_llama",
)
def main(cfg):
    root_dir = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main"
    model_root_dir="/root/autodl-tmp"  # 模型路径
    # TODO:  
    task = "forget01"
    method_name = "RMU"  # ["edit_max", "grad_ascent", "grad_diff", "idk", "npo", "dpo", "ME", "FLAT-TV", "RMU"]
    model_path_for_rmu = "/root/autodl-tmp/tofu_result/llama2-7b/RMU_5e-05_None_layer7_alpha[100.0, 100.0]_coeff_[20.0, 20.0]"
    lr = 5e-05
    num_epochs = 1
    weight_decay = 0.01
    unlearn_dir_name = "tofu_result"
    if cfg.model_family == "phi":
        retain_file_name = "ft_epoch5_lr2e-05_phi_retain" + str(100 - int(task[-2:])).zfill(2) + "_wd0.01"
    elif cfg.model_family == "llama2-7b":
        retain_file_name = "ft_epoch5_lr1e-05_llama2-7b_retain" + str(100 - int(task[-2:])).zfill(2) + "_wd0.01"
    # retain_file_name = "ft_epoch5_lr2e-05_phi_retain" + str(100 - int(task[-2:])).zfill(2) + "_wd0.01"
    # unlearn_file_name = "grad_ascent_1e-05_forget01_5_wd0.01"
    unlearn_file_name = f"{method_name}_{lr}_{task}_{num_epochs}_wd{weight_decay}"
    # unlearn_file_name = "models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd"  # eval originial model
    # "models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd" # eval originial model

    cfg.split = f"{task}_perturbed"

    # evaluate_util.py 部分逻辑
    assert (
        len(cfg.data_path)
        == len(cfg.split_list)
        == len(cfg.eval_task)
        == len(cfg.question_key)
        == len(cfg.answer_key)
        == len(cfg.base_answer_key)
        == len(cfg.perturbed_answer_key)
    ), "data_path, split, eval_task, question_key, and answer_key must be the same length"

    cfg.model_path = f"{model_root_dir}/{unlearn_dir_name}/{cfg.model_family}/{unlearn_file_name}"
    # print("load eval model from:", cfg.model_path)
    if method_name == "RMU":
        cfg.save_dir = f"{root_dir}/{unlearn_dir_name}/{cfg.model_family}/{os.path.basename(model_path_for_rmu)}/ds_size{cfg.ds_size}"
    elif "locuslab--tofu_ft_llama2-7b" in cfg.model_path:
        cfg.save_dir = f"{root_dir}/{unlearn_dir_name}/{cfg.model_family}/locuslab_tofu_ft_llama2-7b/{cfg.split}/ds_size{cfg.ds_size}"
    elif "locuslab--tofu_ft_phi" in cfg.model_path:
        cfg.save_dir = f"{root_dir}/{unlearn_dir_name}/{cfg.model_family}/locuslab_tofu_ft_phi/{cfg.split}/ds_size{cfg.ds_size}"
    else:
        cfg.save_dir = f"{root_dir}/{unlearn_dir_name}/{cfg.model_family}/{unlearn_file_name}/ds_size{cfg.ds_size}"

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    print("save_dir:", cfg.save_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = None
    config = AutoConfig.from_pretrained(model_id)
    for attempt in range(3):
        try:
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=device_map,
                )
            else:
                if method_name == "RMU":
                    print(f"Loading checkpoint from {model_path_for_rmu}")
                    model = AutoModelForCausalLM.from_pretrained(
						model_path_for_rmu,
						config=config,
						use_flash_attention_2=model_cfg["flash_attention2"] == "true",
						torch_dtype=torch.bfloat16,
						# trust_remote_code=True,
						device_map=device_map,
					)
                else:
                    print(f"Loading checkpoint from {cfg.model_path}")
                    model = AutoModelForCausalLM.from_pretrained(
						cfg.model_path,
						config=config,
						use_flash_attention_2=model_cfg["flash_attention2"] == "true",
						torch_dtype=torch.bfloat16,
						trust_remote_code=True,
						device_map=device_map,
					)
        except Exception as e:
            print(e)
            continue
        else:
            break
    else:
        print("Error: could not load model")
    # print(model)
    model = model.eval()
    # =============================== start eval ===============================
    aggregated_eval_logs = {}
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(
        zip(
            cfg.data_path,
            cfg.split_list,
            cfg.question_key,
            cfg.answer_key,
            cfg.eval_task,
            cfg.base_answer_key,
            cfg.perturbed_answer_key,
        )
    ):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"Working on eval task {eval_task} with split {split}")
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = (
            save_filename
            if world_size == 1
            else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")
        )

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
            cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key
        )

        normalize_gt = False
        if "eval_log" not in eval_task:
            normalize_gt = True
        eval_logs = get_all_evals(
            cfg,
            model,
            tokenizer,
            eval_task,
            eval_dataloader,
            base_eval_dataloader,
            perturb_dataloader,
            normalize_gt=normalize_gt,
        )

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)

    # =============================== aggregate ===============================

    retain_result = f"{root_dir}/data/{retain_file_name}/eval_results/ds_size300/eval_log_aggregated.json"
    aggr_save_file = f"{cfg.save_dir}/aggr_result.csv"

    retain_result = json.load(open(retain_result))

    model_utility = get_model_utility(aggregated_eval_logs)
    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
    model_utility["Forget Quality"] = forget_quality["Forget Quality"]

    model_utility["Method"] = method_name
    model_utility["Task"] = task
    if method_name == "RMU":
        model_utility["Comment"] = os.path.basename(model_path_for_rmu)
    else:
        model_utility["Comment"] = unlearn_file_name
    # dump the model utility to a temp.csv
    with open(aggr_save_file, "w") as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)

    return model_utility


if __name__ == "__main__":
    main()
