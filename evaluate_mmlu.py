import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    # config = AutoConfig.from_pretrained(
    #     args.model,
    #     trust_remote_code=True
    # )
    config=None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    if args.subjects is not None:
        subjects=args.subjects
    else:
        subjects = sorted(
            [
                f.split("_test.csv")[0]
                for f in os.listdir(os.path.join(args.data_dir, "test"))
                if "_test.csv" in f
            ]
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model_name))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model_name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model_name), "{}.csv".format(subject)
            ),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        if subcat_cors[subcat]:  # 检查列表是否非空
            try:
                subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
                results["subcategories"][subcat] = subcat_acc
                print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
            except Exception as e:
                print(f"Error calculating accuracy for subcategory {subcat}: {e}")
        else:
            print(f"No data for subcategory {subcat}")

    for cat in cat_cors:
        if cat_cors[cat]:  # 检查列表是否非空
            try:
                cat_acc = np.mean(np.concatenate(cat_cors[cat]))
                results["categories"][cat] = cat_acc
                print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
            except Exception as e:
                print(f"Error calculating accuracy for category {cat}: {e}")
        else:
            print(f"No data for category {cat}")
    
    if all_cors:  # 检查列表是否非空
        weighted_acc = np.mean(np.concatenate(all_cors))
        results["weighted_accuracy"] = weighted_acc
        print("Average accuracy: {:.3f}".format(weighted_acc))
    else:
        print("No data for weighted accuracy calculation")

    results_file = os.path.join(
        args.save_dir, "accuracies_{}.json".format(args.model_name.replace("/", "_"))
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", 
                        default="/home/wxy/wxy_workspace/LLM_unlearn/mmlu_data",
                        type=str)
    parser.add_argument("--save_dir", "-s", 
                        type=str, 
                        default="/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/mmlu_results/")
    parser.add_argument("--model", "-m", 
                        default="/root/autodl-tmp/wmdp_result/zephyr-7b/models--HuggingFaceH4--zephyr-7b-beta/snapshots/892b3d7a7b1cf10c7a701c60881cd93df615734c",
                        type=str)
    parser.add_argument("--model_name", 
                        default="zephyr-7b-beta",
                        type=str)
    # parser.add_argument("--model_type", 
    #                     default="zephyr-7b-beta",
    #                     type=str)
    parser.add_argument("--subjects", 
                        default=["college_biology", "virology", "college_computer_science", "computer_security"],
                        type=list)
    
    args = parser.parse_args()

    args.model = "/root/autodl-tmp/wmdp_result/zephyr-7b/grad_diff_1e-05_sampled-dataset_1_wd0.01"
    args.model_name = "zephyr-7b-beta"
    # args.model_type="microsoft/phi-1_5"
    main(args)
    