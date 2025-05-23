import torch
from torch import nn
from torch.utils.data import Dataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import datasets
import json
from src.utils import get_model_identifiers_from_yaml, add_dataset_index


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token, question_end_token, answer_token = (
        model_configs["question_start_tag"],
        model_configs["question_end_tag"],
        model_configs["answer_tag"],
    )
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded["input_ids"] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded["attention_mask"] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded["input_ids"] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens):
        label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
        
		# TODO:
        # if self.loss_type == "RMU":
        #     # load retain
        #     raw_data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        #     retain_data = []
        #     for x in raw_data:
        #         # if len(x["text"]) > self.min_length:
        #         retain_data.append(str(x["text"]))

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        elif "FLAT" in self.loss_type:
            self.split1, self.split2 = "idk", "forget"
            self.idontknowfile = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            # use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = (
                idx
                if data_type != "retain"
                else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            )
            question = data[idx]["question"]
            answer = data[idx]["answer"]

            if data_type == "idk":
                # get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        split="forget10",
    ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.idontknowfile = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = (
                idx
                if data_type != "retain"
                else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            )

            question = data[idx]["question"]

            if data_type != "idk":
                answer = data[idx]["answer"]
            else:
                # get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            rets.append(converted_data)
        return rets


class WmdpForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, min_length=50, split=None, loss_type="idk"):
        super(WmdpForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        # load forget
        # self.forget_data = datasets.load_dataset(data_path, split)["train"]
        forget_data = []
        if split is not None:
            for line in open(f"{data_path}/{split}.jsonl", "r"):
                if "sampled-dataset" in split or "bio-forget-corpus" in split:
                    raw_text = json.loads(line)["text"]
                else:
                    raw_text = line
                forget_data.append(str(raw_text))
        else:
            for line in open(
                f"{data_path}/bio-forget-corpus.jsonl", "r"
            ):
                raw_text = json.loads(line)["text"]
                if len(raw_text) > self.min_length:
                    forget_data.append(str(raw_text))
            for line in open(
                f"{data_path}/cyber-forget-corpus.jsonl", "r"
            ):
                if len(line) > self.min_length:
                    forget_data.append(str(line))

		# # 随机抽样 3000 到 4000 条数据
        # num_samples = torch.randint(3000, 4001, (1,)).item()
        # sampler = RandomSampler(forget_data, num_samples=num_samples, replacement=False)
        # self.forget_data = [forget_data[i] for i in sampler]
        # print(f"forget data size: {len(self.forget_data)}")	
        self.forget_data = forget_data

        # load retain
        raw_data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        retain_data = []
        for x in raw_data:
            if len(x["text"]) > self.min_length:
                retain_data.append(str(x["text"]))
        self.retain_data = retain_data
        print(f"retain data size: {len(self.retain_data)}")

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        elif "FLAT" in self.loss_type:
            self.split1, self.split2 = "idk", "forget"
            self.idontknowfile = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            # use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = (
                idx
                if data_type != "retain"
                else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            )
            full_text = data[idx]

            if data_type == "idk":
                # get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                full_text = self.idk[rand_pos].strip()

            encoded = self.tokenizer(
                full_text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
            )
            pad_length = self.max_length - len(encoded.input_ids)
            pad_input_ids = encoded["input_ids"] + [self.tokenizer.eos_token_id] * pad_length
            # pad_attention_mask = encoded["attention_mask"] + [0] * pad_length
            # padding_side = left (左填充)
            pad_attention_mask = [0] * pad_length + encoded["attention_mask"] 
            if len(encoded.input_ids) == self.max_length:
                label = encoded.input_ids
            else:
                label = encoded["input_ids"] + [self.tokenizer.eos_token_id] + [-100] * (pad_length - 1)

            converted_data = [torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)]

            rets.append(converted_data)
        return rets



class TextDatasetQA(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
        **kargs,
    ):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]["index"]
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze(),
            torch.tensor(indices),
        )

class TextDatasetWiki(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        min_length=50,
        split=None,
        question_key=None,
        answer_key=None,
        sample_num=5000
    ):
        super(TextDatasetWiki, self).__init__()
        
        raw_data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        retain_data = []
        for x in raw_data:
            if len(x["text"]) > min_length:
                retain_data.append(str(x["text"]))
        self.data = retain_data
        print(f"original retain data size: {len(self.data)}")
        # sample
        if len(self.data) > sample_num:
            self.data = self.data[:sample_num]
        print(f"sampled retain data size: {len(self.data)}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.model_configs["question_start_tag"] = ""
        self.model_configs["question_end_tag"] = ""
        self.model_configs["answer_tag"] = ""
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = ""
        answers = self.data[idx]
        # indices = self.data[idx]["index"]
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze(),
            torch.tensor(idx),
        )


class TextDatasetWMDP(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        min_length=50,
        split=None,
        num_samples=4000,
        **kargs,
    ):
        super(TextDatasetWMDP, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        forget_data = []
        if split is not None:
            # assert(split in ["bio-forget-corpus", "cyber-forget-corpus"], "split must be bio-forget-corpus or cyber-forget-corpus")
            for line in open(f"{data_path}/{split}.jsonl", "r"):
                if "sampled-dataset" in split or "bio-forget-corpus" in split:
                    raw_text = json.loads(line)["text"]
                else:
                    raw_text = line
                forget_data.append(str(raw_text))
        else:
            for line in open(
                f"{data_path}/bio-forget-corpus.jsonl", "r"
            ):
                raw_text = json.loads(line)["text"]
                if len(raw_text) > self.min_length:
                    forget_data.append(str(raw_text))
            for line in open(
                f"{data_path}/cyber-forget-corpus.jsonl", "r"
            ):
                if len(line) > self.min_length:
                    forget_data.append(str(line))

		# 随机抽样 3000 到 4000 条数据
        # num_samples = torch.randint(3000, 4001, (1,)).item()
        sampler = RandomSampler(forget_data, num_samples=num_samples, replacement=False)
        self.data = [forget_data[i] for i in sampler]
        print(f"forget data size: {len(self.data)}")	


        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.model_configs["question_start_tag"] = ""
        self.model_configs["question_end_tag"] = ""
        self.model_configs["answer_tag"] = ""
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = ""
        answers = self.data[idx]
        # indices = self.data[idx]["index"]
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return (
            torch.stack(pad_input_ids_list).squeeze(),
            torch.stack(label_list).squeeze(),
            torch.stack(pad_attention_mask_list).squeeze(),
            torch.tensor(idx),
        )



class TextDatasetQIDK(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        model_family,
        max_length=512,
        split=None,
        question_key="question",
        answer_key="answer",
        **kargs,
    ):
        super(TextDatasetQIDK, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.idontknowfile = "/home/wxy/wxy_workspace/LLM_unlearn/tofu-main/data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        # self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]

        rand_pos = torch.randint(0, len(self.idk), (1,)).item()
        answer = self.idk[rand_pos].strip()

        indices = self.data[idx]["index"]

        converted_data = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length, question, answer, self.model_configs
        )

        return (
            converted_data[0],  # pad_input_ids
            converted_data[1],  # label
            converted_data[2],  # pad_attention_mask
            torch.tensor(indices),
        )


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss
