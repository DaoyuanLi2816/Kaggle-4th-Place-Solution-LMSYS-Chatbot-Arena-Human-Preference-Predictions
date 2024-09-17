# LMSYS - Chatbot Arena Human Preference Predictions 解决方案



### 数据

首先，使用了官方数据(55k) + 33k去重 数据，fold  n_splits=20，只训练了其中一折，尽量保证更多训练数据。其次，为 `ultrafeedback` 数据中的3万条数据做了伪标签，作为更多数据集的补充。



### prompt

设计了独特的prompt，这个prompt的好处是，当对话长度超过max_length时，可以较为合理的截断最后一轮对话，保证了prompt、response_a、response_b 都有一定比例可以展现，避免了最后一轮只会截断到 prompt 或者 response_a 的情况。甚至设定了，如果最后一轮所剩的token数量小于80，会直接丢掉该轮(以及其后面)的对话。这些阈值和比例均通过观察训练集做出的判断。

```python
def tokenize_cls_p3(example, tokenizer, max_length, is_train):
    input_ids = []
    attention_mask = []
    dot_tokens = tokenizer("......", add_special_tokens=False)["input_ids"]
    final_p_tokens = tokenizer("\n\n---\nWhich response is better? [A or B or tie]\nAnswer: ", add_special_tokens=False)["input_ids"]
    for ps, ras, rbs in zip(example['prompt'], example['response_a'], example['response_b']):
        one_input_ids = [tokenizer.bos_token_id]
        prev_tokens_num = 2 + len(final_p_tokens) # 2 for bos_token and eos_token
        for idx, (p, ra, rb) in enumerate(zip(ps, ras, rbs)):
            r_tokens  = tokenizer(f'\n\n## Round {idx+1}:' if idx else f'## Round {idx+1}:', add_special_tokens=False)["input_ids"]
            p_tokens  = tokenizer(f'\n### Prompt:\n{p}', add_special_tokens=False)["input_ids"]
            ra_tokens = tokenizer(f'\n\n### Response A:\n{ra}', add_special_tokens=False)["input_ids"]
            rb_tokens = tokenizer(f'\n\n### Response B:\n{rb}', add_special_tokens=False)["input_ids"]
            all_tokens_num = prev_tokens_num + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens)

            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num - len(r_tokens) - 3*len(dot_tokens) 
                if remain_tokens_num >= 80:
                    p_tokens  =  p_tokens[:int(remain_tokens_num*0.2)] + dot_tokens if len( p_tokens) > int(remain_tokens_num*0.2) else  p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
                    one_input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens
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
```



### 训练

**模型** 

选择`gemma-2-9b-it`，作为起始模型，比其他模型 Llama3 8b 和 Llama3.1 8b 都要好太多了。

使用的是 `Gemma2ForSequenceClassification` 3分类，然后lora bf16对模型进行finetune，最终最高分的实验在4张A100上完成。

max_length ： 2048

**Lora具体参数**

```python
  freeze_layers: 0
  lora_r: 64
  lora_alpha: 16
  lora_dropout: 0.05
  lora_bias: "none"
  lora_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**流程**

1. 第一阶段：使用了官方数据(55k) + 33k去重 数据， fold n_splits=20，并且只训练了其中一折。
2. 第二阶段：使用第一阶段的模型，为 `ultrafeedback` (采样了3万条数据) 做了伪标签，然后与第一阶段的数据合并（总共10万+），从头训练一个模型。

在4*A100 40G上，每次实验，第一阶段花费10小时左右，第二阶段花费15小时左右





### 推理阶段和后处理

推理阶段的代码与训练阶段的代码结构大体上相同，一些不同点在于，在推理阶段将max_length 提升至 3072，其次交换了response_a和response_b 作为tta，最终结果为两者输出的平均值。

针对两种情况进行了后处理（两种情况之间也会有重叠）

1. response_a 或者 response_b 为空，也就是说 类似 '[null]', '[]', '[ ]' 等等这样的情况，理所应当的认为不为空的那个response应该是获胜者，但又因为本次比赛的log loss对于极端值非常敏感，标签又有一定的噪音，所以通过观察训练集将空、不空、平局三者的预测值固定为 [0.04, 0.88, 0.08]。

2. response_a 与 response_b 相同，这种情况理应是平局，所以将预测值固定为 [0.06, 0.06, 0.88]

后处理具体代码如下：

```python
df2 = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
df2['id'] = df2['id'].astype(str)

a_null_df = df2[(df2["response_a"]== '[null]') | (df2["response_a"]== '[]') | (df2["response_a"]== '[ ]') | (df2["response_a"]== '[  ]') | (df2["response_a"]== '[""]') | (df2["response_a"]== '["",""]')]
a_null_id_list = a_null_df["id"].tolist()
submission_df.loc[submission_df['id'].isin(a_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.04, 0.88, 0.08]


b_null_df = df2[(df2["response_b"]== '[null]') | (df2["response_b"]== '[]') | (df2["response_b"]== '[ ]') | (df2["response_b"]== '[  ]') | (df2["response_b"]== '[""]') | (df2["response_b"]== '["",""]')]
b_null_id_list = b_null_df["id"].tolist()
submission_df.loc[submission_df['id'].isin(b_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.88, 0.04, 0.08]


same_a_b_df2 = df2[(df2["response_a"]==df2["response_b"])]
same_a_b_id_list = same_a_b_df2["id"].tolist()
submission_df.loc[submission_df['id'].isin(same_a_b_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.06, 0.06, 0.88]
```



### 总结

**概述**： 开发并优化了一种基于`gemma-2-9b-it`模型的对话系统偏好预测模型，提升了对话系统中用户偏好回复的预测精度。

**关键技术**：

- **数据处理**：使用88k官方和去重数据，20折交叉验证，仅训练一折，并为ultrafeedback数据伪标签扩展至10万+数据集。
- **模型优化**：通过LoRA微调`Gemma2ForSequenceClassification`模型，采用三分类任务，结合独特prompt设计，提升长对话截断处理效果。
- **推理与后处理**：采用TTA策略提升推理效果，并根据空回复和平局情况进行特定后处理，优化模型表现。

