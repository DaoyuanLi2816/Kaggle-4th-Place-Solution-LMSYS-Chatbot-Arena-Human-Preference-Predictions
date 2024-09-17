
# LMSYS - Chatbot Arena Human Preference Prediction Solution

### Data

First, we used the official dataset (55k) combined with 33k de-duplicated data, applying a 20-fold cross-validation (n_splits=20), but only trained one fold to ensure the inclusion of as much training data as possible. Additionally, we created pseudo-labels for 30k entries from the `ultrafeedback` dataset, serving as an augmentation to the dataset.

### Prompt

We designed a unique prompt. The advantage of this prompt is that when the conversation length exceeds the maximum token limit (`max_length`), it can reasonably truncate the last round of the dialogue. This ensures that the prompt, `response_a`, and `response_b` all have a proportional representation, avoiding scenarios where only the prompt or `response_a` is truncated. We also set a rule: if the remaining token count in the last round is less than 80, we discard that round (and any following dialogues). These thresholds and ratios were determined based on observations from the training set.

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
            r_tokens = tokenizer(f'\n\n## Round {idx+1}:' if idx else f'## Round {idx+1}:', add_special_tokens=False)["input_ids"]
            p_tokens = tokenizer(f'\n### Prompt:\n{p}', add_special_tokens=False)["input_ids"]
            ra_tokens = tokenizer(f'\n\n### Response A:\n{ra}', add_special_tokens=False)["input_ids"]
            rb_tokens = tokenizer(f'\n\n### Response B:\n{rb}', add_special_tokens=False)["input_ids"]
            all_tokens_num = prev_tokens_num + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens)

            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num - len(r_tokens) - 3*len(dot_tokens)
                if remain_tokens_num >= 80:
                    p_tokens = p_tokens[:int(remain_tokens_num*0.2)] + dot_tokens if len(p_tokens) > int(remain_tokens_num*0.2) else p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num*0.4)] if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num*0.4)] if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
                else:
                    continue
            one_input_ids.extend(r_tokens + p_tokens + ra_tokens + rb_tokens)
        one_input_ids.append(tokenizer.eos_token_id)
        input_ids.append(one_input_ids)
        attention_mask.append([1]*len(one_input_ids))
    return input_ids, attention_mask
```

### Model

We fine-tuned the `Gemma2ForSequenceClassification` model using LoRA on a three-way classification task, with the categories being `response_a` better, `response_b` better, or a tie. We did not apply any tricks in this phase, merely leveraging the designed prompt to ensure the appropriate truncation of long dialogues.

In the second phase, we created pseudo-labels for the `ultrafeedback` data (30k samples), then combined this with the first-phase data (over 100k in total) to train a model from scratch.

Each experiment took approximately 10 hours in the first phase and 15 hours in the second phase on a 4x A100 40G setup.

### Inference and Post-processing

The inference code is largely the same as the training code, with some differences. During inference, we raised `max_length` to 3072 and swapped `response_a` and `response_b` for test-time augmentation (TTA), averaging the results from both configurations.

We applied post-processing in two scenarios (there is some overlap between the two):

1. If `response_a` or `response_b` is empty (e.g., '[null]', '[]', '[ ]'), we assumed the non-empty response was the winner. However, due to the sensitivity of the log loss to extreme values and the noise in the labels, we fixed the predictions for empty, non-empty, and tie situations to [0.04, 0.88, 0.08] based on observations from the training set.

2. If `response_a` and `response_b` are identical, we assumed it was a tie and fixed the prediction values to [0.06, 0.06, 0.88].

The post-processing code is as follows:

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

### Summary

**Overview**: We developed and optimized a chatbot preference prediction model based on the `gemma-2-9b-it` model, improving the prediction accuracy of user preferences in chatbot responses.

**Key Techniques**:

- **Data Processing**: Used 88k official and de-duplicated data, performed 20-fold cross-validation, trained only one fold, and created pseudo-labels for the `ultrafeedback` dataset, expanding it to over 100k samples.
- **Model Optimization**: Fine-tuned the `Gemma2ForSequenceClassification` model using LoRA, applied a unique prompt design, and improved handling of long dialogue truncations.
- **Inference and Post-processing**: Employed TTA to enhance inference results and applied specific post-processing for empty responses and ties, optimizing the model's performance.
