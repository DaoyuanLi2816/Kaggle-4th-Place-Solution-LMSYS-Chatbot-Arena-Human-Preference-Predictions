# %% ========= Import and Config =========
import os
import copy
from dataclasses import dataclass
import argparse
import numpy as np
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
parser.add_argument('--cfg', type=str, default='train_v107.yaml')
parser.add_argument('--diff_cfg', type=str, default='', help='diff config file. like: two_spec_022401.yaml')
args = parser.parse_args()

cfg = load_yaml(args.cfg)
cfg = simple_namespace(cfg)

base_dir = "."
input_dir = f"{base_dir}/input"
output_dir = f"{base_dir}/output"
summary_log_path = f"{output_dir}/summary.log"


seed_everything(cfg.general.seed)
cur_time = current_date_time()
cur_time_abbr = cur_time.replace("-", "").replace(":", "").replace(" ", "")[4:12]
output_dir = f"{output_dir}/{cur_time_abbr}"
os.makedirs(output_dir, exist_ok=True)
LOGGER = init_logger(f'{output_dir}/train.log')
shutil.copy(args.cfg, f"{output_dir}/{args.cfg}")

real_diff_file, differences = compare_yaml(args.diff_cfg, args.cfg)
formatted_diffs = format_diffs(differences)
base_info = f"\n========== {cur_time_abbr} - {args.cfg} ==========\nCompare with: {real_diff_file}\n{formatted_diffs}"
write_to_summary_log(summary_log_path,  base_info)
LOGGER.info(base_info)

num_gpus = torch.cuda.device_count()
LOGGER.info(f"可用的 GPU 数量: {num_gpus}")

# %% ========= Read Data =========
if os.path.exists(cfg.data.parquet_path):
    LOGGER.info(f"Reading data from {cfg.data.parquet_path}")
    df = pd.read_parquet(cfg.data.parquet_path)
else:
    LOGGER.info(f"Reading data from csv files...")
    if cfg.data.use_55k:
        df_55k = pd.read_csv(f'{input_dir}/lmsys-chatbot-arena/train.csv', encoding='utf-8')
        df_55k = df_55k[~((df_55k["response_a"]== '[null]') & (df_55k["response_b"]== '[null]'))] # 去掉response_a和response_b都是null的数据
        df_55k = df_55k[~(df_55k["response_a"]==df_55k["response_b"])] # 去掉response_a和response_b相同的数据

        a_null_df = df_55k[(df_55k["response_a"]== '[null]') | (df_55k["response_a"]== '[]') | (df_55k["response_a"]== '[ ]') | (df_55k["response_a"]== '[  ]') | (df_55k["response_a"]== '[""]') | (df_55k["response_a"]== '["",""]')]
        a_null_id_list = a_null_df["id"].tolist()
        df_55k.loc[df_55k['id'].isin(a_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.0, 1.0, 0.0]

        b_null_df = df_55k[(df_55k["response_b"]== '[null]') | (df_55k["response_b"]== '[]') | (df_55k["response_b"]== '[ ]') | (df_55k["response_b"]== '[  ]') | (df_55k["response_b"]== '[""]') | (df_55k["response_b"]== '["",""]')]
        b_null_id_list = b_null_df["id"].tolist()
        df_55k.loc[df_55k['id'].isin(b_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [1.0, 0.0, 0.0]

        df_55k['prompt'] = df_55k['prompt'].apply(process)
        df_55k['response_a'] = df_55k['response_a'].apply(process)
        df_55k['response_b'] = df_55k['response_b'].apply(process)

    if cfg.data.use_33k:
        df_33k = pd.read_csv(f'{input_dir}/lmsys-33k/lmsys-33k-deduplicated.csv', encoding='utf-8')
        df_33k = df_33k[~(df_33k["response_a"]==df_33k["response_b"])] # 去掉response_a和response_b相同的数据

        a_null_df = df_33k[(df_33k["response_a"]== '[null]') | (df_33k["response_a"]== '[]') | (df_33k["response_a"]== '[ ]') | (df_33k["response_a"]== '[  ]') | (df_33k["response_a"]== '[""]') | (df_33k["response_a"]== '["",""]')]
        a_null_id_list = a_null_df["id"].tolist()
        df_33k.loc[df_33k['id'].isin(a_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.0, 1.0, 0.0]

        b_null_df = df_33k[(df_33k["response_b"]== '[null]') | (df_33k["response_b"]== '[]') | (df_33k["response_b"]== '[ ]') | (df_33k["response_b"]== '[  ]') | (df_33k["response_b"]== '[""]') | (df_33k["response_b"]== '["",""]')]
        b_null_id_list = b_null_df["id"].tolist()
        df_33k.loc[df_33k['id'].isin(b_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [1.0, 0.0, 0.0]

        df_33k['prompt'] = df_33k['prompt'].apply(process)
        df_33k['response_a'] = df_33k['response_a'].apply(process)
        df_33k['response_b'] = df_33k['response_b'].apply(process)

    if cfg.data.use_55k and cfg.data.use_33k:
        df = pd.concat([df_55k, df_33k], axis=0)
    elif cfg.data.use_55k:
        df = df_55k
    elif cfg.data.use_33k:
        df = df_33k
    
    df = df.reset_index(drop=True)
    df['id'] = df['id'].astype(str)

    df.to_parquet(cfg.data.parquet_path)

if cfg.data.fixed_val_ids:
    with open(f"{input_dir}/val_ids_n{cfg.data.n_splits}.json", "r") as f:
        val_ids = json.load(f)

    train_idx = df[~df['id'].isin(val_ids)].index
    val_idx = df[df['id'].isin(val_ids)].index
    
else:
    kf = KFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.general.seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        if fold == cfg.data.fold_idx:
            break

    val_ids = df.loc[val_idx, 'id'].values.tolist()
    with open(f"{output_dir}/val_ids_n{cfg.data.n_splits}.json", "w") as f:
        json.dump(val_ids, f)

LOGGER.info(f"train_idx: {len(train_idx)}")
LOGGER.info(f"val_idx: {len(val_idx)}\n\n")

LOGGER.info(f"df.shape: {df.shape}\n")
df.head()

# %% ========= Tokenizer and Dataset ========= 
tokenizer = GemmaTokenizerFast.from_pretrained(f"{input_dir}/{cfg.model.model_name}")
tokenizer.add_eos_token = True
tokenizer.padding_side = cfg.data.padding_side
tokenizer.truncation_side = cfg.data.truncation_side

def tokenize_cls_p3(example, tokenizer, max_length):
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

    labels = [0 if a_win else 1 if b_win else 2 for a_win, b_win, tie in zip(example['winner_model_a'], example['winner_model_b'], example['winner_tie'])]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

ds = Dataset.from_pandas(df)
ds = ds.map(
    eval(cfg.data.tokenzier_prompt_func), 
    batched=True, 
    num_proc=cfg.general.num_workers,
    remove_columns=["id", "model_a", "model_b", "prompt", "response_a", "response_b", "winner_model_a", "winner_model_b", "winner_tie"], 
    fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.model.max_length,}
    )

LOGGER.info(f"tokenzier_prompt_func: {cfg.data.tokenzier_prompt_func}\n")
LOGGER.info(f"len(ds): {len(ds)}\n")
LOGGER.info("================================================== example ==================================================")
LOGGER.info(f"input_ids length: {len(ds[0]['input_ids'])}\n")
LOGGER.info(tokenizer.decode(ds[0]['input_ids']))
LOGGER.info("=============================================================================================================\n\n")

# %% ========= Model =========
layers_num = 42

lora_config = LoraConfig(
    r=cfg.model.lora_r,
    lora_alpha=cfg.model.lora_alpha,
    lora_dropout=cfg.model.lora_dropout,
    bias=cfg.model.lora_bias,
    task_type=TaskType.SEQ_CLS,
    target_modules=cfg.model.lora_target_modules,
    # layers_to_transform=[i for i in range(layers_num) if i >= cfg.model.freeze_layers],
)

model = Gemma2ForSequenceClassification.from_pretrained(
    f"{input_dir}/{cfg.model.model_name}",
    num_labels=3,
    torch_dtype=torch.bfloat16 if cfg.training.amp == "bf16" else torch.float16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)


model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
if not cfg.model.use_softcapping:
    model.config.attn_logit_softcapping = None
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()
LOGGER.info(f"model:\n{model}\n\n")

# %% ========= Training =========
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    dataloader_num_workers=cfg.general.num_workers,
    report_to="none",
    num_train_epochs=cfg.training.n_epochs,
    per_device_train_batch_size=cfg.training.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
    logging_steps=50,
    logging_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=2000,
    optim=cfg.training.optim_type,
    fp16=True if cfg.training.amp == "fp16" else False,
    bf16=True if cfg.training.amp == "bf16" else False,
    learning_rate=cfg.training.lr,
    lr_scheduler_type='cosine',
    warmup_ratio = 0.1,
    warmup_steps=cfg.training.warmup_steps,
    metric_for_best_model="log_loss",
    greater_is_better=False,
)

LOGGER.info(f"Training arguments: {training_args}\n\n")

trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(val_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# 开始训练
LOGGER.info("start training...")
trainer.train()

# 保存模型
trainer.save_model(f"{output_dir}/{cur_time_abbr}adapetermodel")
LOGGER.info("finish training...")

# 合并, 保存完整模型
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{output_dir}/{cur_time_abbr}mergedmodel")
tokenizer.save_pretrained(f"{output_dir}/{cur_time_abbr}mergedmodel")


# 记录 eval_loss
eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
if eval_losses:
    last_eval_loss = eval_losses[-1]
    write_to_summary_log(summary_log_path,  f"Last eval_loss: {last_eval_loss}")
    LOGGER.info(f"Last eval_loss: {last_eval_loss}")
else:
    LOGGER.info("No eval_loss found in log history.")