import sys
import os

sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main")
sys.path.append("/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/src")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import hydra
import transformers
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from src.utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
from time import time
import json

from src.data_module import TextDatasetQA, TextDatasetQIDK
from src.dataloader import custom_data_collator_forget, custom_data_collator_forget_dpo
from src.trainer import CustomTrainerForgetting
from src.AlphaEdit_hparams import AlphaEditHyperParams
from src.AlphaEdit_main import apply_AlphaEdit_to_model

dataset_class_mapping = {
    "tofu": TextDatasetQA,
    "tofu_idk": TextDatasetQIDK,
    "wmdp": None,
}
ROOT_DIR = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main"


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
    param_ids = [i for i in range(20)]
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


# def chunks(arr, n):
#     """Yield successive n-sized chunks from arr."""
#     for i in range(0, len(arr), n):
#         yield arr[i : i + n]
def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for i in range(len(arr)):
        chunk.append(arr[i])
        if len(chunk) == n or i == len(arr) - 1:
            yield chunk
            chunk = []


@hydra.main(
    version_base=None,
    config_path=f"{ROOT_DIR}/config/forget",
    config_name="forget_phi_tofu",
)
def main(cfg):
    # TODO: 下面代码最好改成读取配置文件的方式，先放着，跑通再改
    cfg.lr = 1e-5
    cfg.weight_decay = 0.01
    cfg.batch_size = 40
    # cfg.gradient_accumulation_steps = 8
    cfg.num_epochs = 1
    cfg.data_name = "tofu"  # tofu, wmdp, tofu_idk
    cfg.split = "forget01" if "tofu" in cfg.data_name else None  #
    retain_split = "retain" + str(100 - int(cfg.split[-2:])).zfill(2) if "tofu" in cfg.data_name else None
    cfg.forget_loss = "edit_max"  # ["grad_ascent", "grad_diff", "idk", "npo", "dpo", "ME", "FLAT-TV", "RMU"]
    cfg.save_dir = f"{ROOT_DIR}/{cfg.data_name}_result/{cfg.forget_loss}_{cfg.lr}_{cfg.split}_{cfg.num_epochs}_wd{cfg.weight_decay}_bs{cfg.batch_size}"

    params_path = f"{ROOT_DIR}/config/edit/{cfg.model_family}.json"
    hparams = AlphaEditHyperParams.from_json(params_path)
    print(f"Executing with parameters {hparams}")

    layers_name = "".join(str(i) for i in hparams.layers)
    null_space_proj_path = f"{ROOT_DIR}/data/stats/{cfg.model_family}/{cfg.data_name[:4]}_{retain_split}_{layers_name}_null_space_project.pt"
    cache_k0_path = (
        f"{ROOT_DIR}/data/stats/{cfg.model_family}/{cfg.data_name[:4]}_{retain_split}_{layers_name}_cache_k0.pt"
    )
    cache_template = (
        f"{ROOT_DIR}/data/kvs/{cfg.model_family}/{cfg.data_name}_{cfg.split}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
    )

    max_length = 512  # tokenizer.model_max_length
    num_edits = cfg.batch_size

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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # =============================== load data ===============================

    torch_format_dataset = dataset_class_mapping[cfg.data_name](
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=max_length,
        split=cfg.split,
        question_key="question",
        answer_key="answer",
    )

    # =============================== load model ===============================
    try:
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
    except:
        print("Error! Model not found in the path")
        return

    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    # =============================== config ===============================
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

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference

    # =============================== start train() ===============================
    if os.path.exists(null_space_proj_path):
        print("Loading null space projection from", null_space_proj_path)
        P = torch.load(null_space_proj_path).cpu()
        cache_c = torch.zeros_like(P).cpu()
    else:
        print("Null space projection not found.")
        return

    cache_k0 = None
    if "k0" in cfg.forget_loss:
        if os.path.exists(cache_k0_path):
            print("Loading cached K0 from", cache_k0_path)
            cache_k0 = torch.load(cache_k0_path).cpu()
        else:
            print("Cached K0 not found.")
            return
    cnt = 0

    # case_result_template = str(cfg.save_dir + "/{}_edits-case_{}.json")
    for data_chunks in chunks(torch_format_dataset, num_edits):  # list of data
        print(f"========================================{cnt+1}_edit==================================")
        # Compute weight changes + record weights that changed
        args_conserve_memory = dict()
        etc_args = dict(cache_template=cache_template)
        seq_args = dict(cache_c=cache_c, cache_k=cache_k0)
        nc_args = dict(P=P)
        edited_model, cache_c = apply_AlphaEdit_to_model(
            model,
            tokenizer,
            [
                {
                    "case_id": data[3],
                    "input_ids": data[0].to("cuda"),
                    "labels": data[1].to("cuda"),
                    "attention_mask": data[2].to("cuda"),
                }
                for data in data_chunks
            ],
            hparams,
            cfg,
            **args_conserve_memory,
            **etc_args,
            **seq_args,
            **nc_args,
        )
        cnt += 1
    # =============================== save result ===============================
    # save the tokenizer
    if cfg.save_model:
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                # delete the directory
                import shutil

                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
