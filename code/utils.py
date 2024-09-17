import json
import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import random
import time
from torch import nn
import random
import os
from datetime import datetime
import math
import yaml
from deepdiff import DeepDiff
from types import SimpleNamespace
from glob import glob
from sklearn.metrics import log_loss, accuracy_score


def get_now_time_fullstring():
    # 返回当前时间的完整字符串格式，例如 "2023-06-05 16:35:00"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def current_date_time():
    # Format the current date and time as "YYYY-MM-DD HH:MM:SS"
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def seed_everything(seed=42):
    # 设置随机种子，以确保实验的可重复性
    # seed 是一个整数，表示随机种子
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

def init_logger(log_file):
    # 初始化日志器
    # log_file 是一个字符串，表示日志文件的路径
    
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def get_timediff(time1,time2):
    # 计算两个时间点之间的时间差，并以分钟和秒的形式返回
    # time1 和 time2 是两个时间点，以秒为单位
    
    minute_,second_ = divmod(time2-time1,60)
    return f"{int(minute_):02d}:{int(second_):02d}"

def write_to_summary_log(summary_log_file, message):
    with open(summary_log_file, 'a+') as file:
        file.write(f"{message}\n")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def simple_namespace(cfg):
    for k, v in cfg.items():
        if type(v) == dict:
            cfg[k] = SimpleNamespace(**v)
    return SimpleNamespace(**cfg)

def compare_yaml(file1, file2):
    '''
    Compare two yaml files and return the differences
    '''
    # 如果没有指定file1, 那么就和file2上一个yaml文件进行比较
    if not file1:
        all_yaml_files = sorted(glob("*.yaml"))
        if all_yaml_files.index(file2) == 0:
            print("No previous yaml file found.")
            file1 = file2
        else:
            file1 = all_yaml_files[all_yaml_files.index(file2)-1]

    yaml1 = load_yaml(file1)
    yaml2 = load_yaml(file2)
    
    def get_value_from_path(data, path):
        elements = path.strip("root").strip("[").strip("]").replace("'", "").split('][')
        for element in elements:
            try:
                if element.isdigit():
                    data = data[int(element)]
                else:
                    data = data[element]
            except (KeyError, TypeError, IndexError):
                return None
        return data

    diff = DeepDiff(yaml1, yaml2, ignore_order=True)

    # Enhance diff with actual values for added and removed items
    added_values = {}
    removed_values = {}
    for path in diff.get('dictionary_item_added', []):
        value = get_value_from_path(yaml2, path)
        added_values[path] = value

    for path in diff.get('dictionary_item_removed', []):
        value = get_value_from_path(yaml1, path)
        removed_values[path] = value

    if added_values:
        diff['dictionary_item_added_values'] = added_values
    if removed_values:
        diff['dictionary_item_removed_values'] = removed_values

    return file1, diff



def format_diffs(diffs):
    '''
    格式化diffs
    '''
    formatted_diffs = ""
    
    # 处理值变化
    for diff_type, changes in diffs.items():
        if diff_type == 'values_changed':
            for key, value in changes.items():
                path = key.split('[')[1:]
                path = [p.strip("]'") for p in path]
                path_str = " - ".join(path)
                formatted_diffs += f"{path_str}: {value['old_value']} --> {value['new_value']}\n"
        
        # 处理添加的项目
        elif diff_type == 'dictionary_item_added_values':
            for key, value in changes.items():
                path = key.split('[')[1:]
                path = [p.strip("]'") for p in path]
                path_str = " - ".join(path)
                formatted_diffs += f"[add] {path_str}: {value}\n"
        
        # 处理删除的项目
        elif diff_type == 'dictionary_item_removed_values':
            for key, value in changes.items():
                path = key.split('[')[1:]
                path = [p.strip("]'") for p in path]
                path_str = " - ".join(path)
                formatted_diffs += f"[remove] {path_str}: {value}\n"

    return formatted_diffs


def get_parameter_number(model, unit='M'):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    div = {"M": 1e6, "K": 1e3, "B": 1}[unit]
    return f'Total params: {total_num/div:.1f}{unit}; Trainable params: {trainable_num//div:.1f}{unit}'




def process(input_str):
    sentences = json.loads(input_str)
    return sentences


def compute_metrics(eval_preds) -> dict:
    preds = eval_preds.predictions # (n, 3)
    labels = eval_preds.label_ids # (n, 3) one-hot
    if labels.shape[-1] == 3:
        labels = labels.argmax(-1)
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}