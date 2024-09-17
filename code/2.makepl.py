# %% ========= Import and Config =========
import os
import copy
from dataclasses import dataclass
import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,

    Gemma2PreTrainedModel,
    Gemma2Model,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,

    PreTrainedTokenizerFast,
    PreTrainedTokenizerBase, 
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import pandas as pd
import json
import shutil
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.special import softmax

# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'

from utils import get_now_time_fullstring, seed_everything, current_date_time
from utils import load_yaml, simple_namespace, compare_yaml, format_diffs, write_to_summary_log, init_logger
from utils import process, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./output/08040309/train_v106.yaml')
args = parser.parse_args()

cfg = load_yaml(args.cfg)
cfg = simple_namespace(cfg)

SUFFIX = "08040309"
MAX_LEN = 3072

base_dir = "."
input_dir = f"{base_dir}/input"
output_dir = f"{base_dir}/output/{SUFFIX}"

WEIGHTS_PATH = f"{output_dir}/merged_model"

seed_everything(cfg.general.seed)
cur_time = current_date_time()
cur_time_abbr = cur_time.replace("-", "").replace(":", "").replace(" ", "")[4:12]
LOGGER = init_logger(f'{output_dir}/makepl_{cur_time_abbr}.log')

num_gpus = torch.cuda.device_count()
LOGGER.info(f"可用的 GPU 数量: {num_gpus}")


# %% ========= Load Data =========
df = pd.read_parquet(f"{input_dir}/ultrafeedback.parquet")

df["prompt"] = df["prompt"].apply(lambda x: [x])
df["chosen"] = df["chosen"].apply(lambda x: [x[1]["content"]])
df["rejected"] = df["rejected"].apply(lambda x: [x[1]["content"]])

data = []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    if random.random() > 0.5:
        response_a = row['chosen']
        response_b = row['rejected']
        winner = 0
    else:
        response_a = row['rejected']
        response_b = row['chosen']
        winner = 1
    
    data.append({
        'prompt': row['prompt'],
        'response_a': response_a,
        'response_b': response_b,
        'winner': winner
    })

df = pd.DataFrame(data)
df["tmp_prompt"] = df["prompt"].apply(lambda x: x[0])
df = df.drop_duplicates(subset=['tmp_prompt'], ignore_index=True)
df = df.drop(columns=['tmp_prompt'])
df['id'] = df.index
df = df.sort_values("id", ascending=True).reset_index(drop=True)
LOGGER.info(f"df.shape: {df.shape}\n")


# %% ========= Tokenizer and Dataset ========= 
tokenizer = GemmaTokenizerFast.from_pretrained(WEIGHTS_PATH)
tokenizer.add_eos_token = True
tokenizer.padding_side = cfg.data.padding_side
tokenizer.truncation_side = cfg.data.truncation_side

def tokenize_cls_p3(example, tokenizer, max_length, is_train):
    input_ids = []
    attention_mask = []
    dot_tokens = tokenizer("......", add_special_tokens=False)["input_ids"]
    final_p_tokens = tokenizer("\n\n---\nWhich response is better? [A or B or tie]\nAnswer: ", add_special_tokens=False)["input_ids"]
    for ps, ras, rbs in zip(example['prompt'], example['response_a'], example['response_b']):
        one_input_ids = [tokenizer.bos_token_id] # 一个样本的所有tokens
        prev_tokens_num = 2 + len(final_p_tokens) # 2 for bos_token and eos_token
        for idx, (p, ra, rb) in enumerate(zip(ps, ras, rbs)):
            r_tokens  = tokenizer(f'\n\n## Round {idx+1}:' if idx else f'## Round {idx+1}:', add_special_tokens=False)["input_ids"] # 对于 Round 1, 前面不需要换行
            p_tokens  = tokenizer(f'\n### Prompt:\n{p}', add_special_tokens=False)["input_ids"]
            ra_tokens = tokenizer(f'\n\n### Response A:\n{ra}', add_special_tokens=False)["input_ids"]
            rb_tokens = tokenizer(f'\n\n### Response B:\n{rb}', add_special_tokens=False)["input_ids"]
            all_tokens_num = prev_tokens_num + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens)

            # 如果 加上当前轮的tokens 超过了 max_length
            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num - len(r_tokens) - 3*len(dot_tokens)  # 剩余可分配的 p,a,b token数量
                if remain_tokens_num >= 80:
                    # 可分配的 p,a,b token数量 > 80, 可以对 p,a,b token 进行截断
                    p_tokens  =  p_tokens[:int(remain_tokens_num*0.2)] + dot_tokens if len( p_tokens) > int(remain_tokens_num*0.2) else  p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
                    one_input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens # 添加到 input_ids
                break
            else:
                prev_tokens_num = all_tokens_num
                one_input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens
        
        one_input_ids += final_p_tokens + [tokenizer.eos_token_id]
        one_attention_mask = [1] * len(one_input_ids)

        input_ids.append(one_input_ids)
        attention_mask.append(one_attention_mask)
    
    if is_train:
        labels = [0 if a_win else 1 if b_win else 2 for a_win, b_win, tie in zip(example['winner_model_a'], example['winner_model_b'], example['winner_tie'])]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def infer_process_batch(df, tokenizer, max_length, batch_size):
    results = {"input_ids": [], "attention_mask": []}

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        example = {
            'prompt': batch['prompt'].tolist(),
            'response_a': batch['response_a'].tolist(),
            'response_b': batch['response_b'].tolist(),
        }
        tokenized = tokenize_cls_p3(example, tokenizer, max_length, is_train=False)
        results["input_ids"].extend(tokenized["input_ids"])
        results["attention_mask"].extend(tokenized["attention_mask"])
        
    return results


tokenized_results = infer_process_batch(
    df=df, 
    tokenizer=tokenizer, 
    max_length=MAX_LEN, 
    batch_size=100
)

LOGGER.info(f"len: {len(tokenized_results['input_ids'])}\n\n")
LOGGER.info(f"preview decode(input_ids[0]):\n==========\n{tokenizer.decode(tokenized_results['input_ids'][0])}\n==========\n\n")
LOGGER.info(f"preview input_ids[0]:\n==========\n{tokenized_results['input_ids'][0]}\n==========\n\n")
LOGGER.info(f"preview attention_mask[0]:\n==========\n{tokenized_results['attention_mask'][0]}\n==========\n\n")


data = pd.DataFrame()
data["id"] = df["id"]
data['input_ids'] = tokenized_results['input_ids']
data['attention_mask'] = tokenized_results['attention_mask']
data["length"] = data["input_ids"].apply(len)


# %% ========= Model =========
bnb_config =  BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)

base_model = Gemma2ForSequenceClassification.from_pretrained(
    WEIGHTS_PATH,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='auto')

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.use_cache = False
if not cfg.model.use_softcapping:
    base_model.config.attn_logit_softcapping = None

LOGGER.info(f"model:\n{base_model}\n\n")

# %% ========= Inference =========
@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=2, max_length=MAX_LEN):
    a_win, b_win, tie = [], [], []
    
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        
        proba = outputs.logits.softmax(-1).cpu()
        
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    
    return df

LOGGER.info(f"Start Inference...")
data = data.sort_values("length", ascending=False)
results = inference(data, base_model, torch.device("cuda"))


results_df = results.sort_values("id", ascending=True).reset_index(drop=True)
# 加入 df 中的 prompt, response_a, response_b
results_df["prompt"] = df["prompt"]
results_df["response_a"] = df["response_a"]
results_df["response_b"] = df["response_b"]

results_df.to_csv(f"{input_dir}/ultrafeedback_pl.csv", index=False)
