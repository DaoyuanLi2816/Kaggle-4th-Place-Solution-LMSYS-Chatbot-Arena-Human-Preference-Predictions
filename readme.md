
# LMSYS - Chatbot Arena Human Preference Prediction Solution

This solution was developed for the [LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview) competition on Kaggle, where participants were challenged to predict user preferences in head-to-head conversations between chatbots powered by large language models (LLMs). The task involved utilizing a dataset from **Chatbot Arena**, in which users interact with two anonymous LLMs and choose their preferred response. By creating a machine learning model that accurately predicts these preferences, we aimed to contribute to improving the alignment of chatbot responses with human preferences.

Our team successfully placed **4th out of 1849 teams**, earning a **Gold Medal** for our solution! 🏅

![Daoyuan Li - LMSYS](./Daoyuan%20Li%20-%20LMSYS%20-%20Chatbot%20Arena%20Human%20Preference%20Predictions.png)

## Data
First, we utilized the official dataset (55k) along with 33k deduplicated data, employing a 20-fold cross-validation (n_splits=20), but only trained on one fold to maximize the amount of training data. Additionally, we created pseudo-labels for 30,000 entries from the ultrafeedback dataset to further supplement the dataset.

## Prompt
We designed a unique prompt, which is beneficial because when the dialogue length exceeds the maximum token length (`max_length`), it allows for a reasonable truncation of the final round of conversation. This ensures that the prompt, response A, and response B can all be adequately displayed, avoiding situations where only the prompt or response A gets truncated. If the remaining token count in the final round is less than 80, the entire conversation round (and the subsequent ones) will be discarded. Th...

```python
def tokenize_cls_p3(example, tokenizer, max_length, is_train):
    input_ids = []
    attention_mask = []
    dot_tokens = tokenizer("......", add_special_tokens=False)["input_ids"]
    final_p_tokens = tokenizer("\n\n---\nWhich response is better? [A or B or tie]\nAnswer: ", add_special_tokens=False)["input_ids"]

    for ps, ras, rbs in zip(example['prompt'], example['response_a'], example['response_b']):
        one_input_ids = [tokenizer.bos_token_id]
        prev_tokens_num = 2 + len(final_p_tokens)  # 2 for bos_token and eos_token

        for idx, (p, ra, rb) in enumerate(zip(ps, ras, rbs)):
            r_tokens = tokenizer(f'\n\n## Round {idx+1}:' if idx else f'## Round {idx+1}:', add_special_tokens=False)["input_ids"]
            p_tokens = tokenizer(f'\n### Prompt:\n{p}', add_special_tokens=False)["input_ids"]
            ra_tokens = tokenizer(f'\n\n### Response A:\n{ra}', add_special_tokens=False)["input_ids"]
            rb_tokens = tokenizer(f'\n\n### Response B:\n{rb}', add_special_tokens=False)["input_ids"]

            all_tokens_num = prev_tokens_num + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens

            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num - len(r_tokens) - 3 * len(dot_tokens)
                if remain_tokens_num >= 80:
                    p_tokens = p_tokens[:int(remain_tokens_num * 0.2)] + dot_tokens if len(p_tokens) > int(remain_tokens_num * 0.2) else p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num * 0.4)] + dot_tokens if len(ra_tokens) > int(remain_tokens_num * 0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num * 0.4)] + dot_tokens if len(rb_tokens) > int(remain_tokens_num * 0.4) else rb_tokens

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
## Training

### Model
We selected **gemma-2-9b-it** as the starting model, which significantly outperforms other models such as **Llama3 8b** and **Llama3.1 8b**. We used **Gemma2ForSequenceClassification** for a three-class classification task, and fine-tuned the model using **lora** with **bf16** precision. The best experimental results were achieved on four A100 GPUs.

- **max_length**: 2048
- LoRA-specific parameters:
  - freeze_layers: 0
  - lora_r: 64
  - lora_alpha: 16
  - lora_dropout: 0.05
  - lora_bias: "none"
  - lora_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

## Process
1. **Phase 1**: We used the official dataset (55k) along with 33k deduplicated data, employing 20-fold cross-validation, but only trained one fold.
2. **Phase 2**: Using the model from the first phase, we generated pseudo-labels for 30,000 entries from the ultrafeedback dataset. These were then merged with the Phase 1 dataset, totaling over 100,000 entries. A new model was trained from scratch.

Each experiment took approximately 10 hours for the first phase and 15 hours for the second phase on a system with 4 A100 GPUs (40G).

## Inference and Post-Processing

The inference phase uses a similar code structure to the training phase, with some key differences: the `max_length` is increased to 3072, and **response_a** and **response_b** are swapped as part of a test-time augmentation (TTA) strategy. The final result is the average output of both.

Post-processing was applied for two specific scenarios (which may overlap):
1. If **response_a** or **response_b** is empty (e.g., '[null]', '[]', '[ ]'), we assume the non-empty response is the winner. However, since the log loss in this competition is very sensitive to extreme values and there is some noise in the labels, we observed the training set and fixed the predictions for empty, non-empty, and tie cases to [0.04, 0.88, 0.08].
2. If **response_a** and **response_b** are identical, we assume a tie and set the prediction to [0.06, 0.06, 0.88].

```python
df2 = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
df2['id'] = df2['id'].astype(str)

a_null_df = df2[(df2["response_a"] == '[null]') | (df2["response_a"] == '[]') | (df2["response_a"] == '[ ]') | (df2["response_a"] == '[  ]') | (df2["response_a"] == '[""]') | (df2["response_a"] == '["",""]')]
a_null_id_list = a_null_df["id"].tolist()
submission_df.loc[submission_df['id'].isin(a_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.04, 0.88, 0.08]

b_null_df = df2[(df2["response_b"] == '[null]') | (df2["response_b"] == '[]') | (df2["response_b"] == '[ ]') | (df2["response_b"] == '[  ]') | (df2["response_b"] == '[""]') | (df2["response_b"] == '["",""]')]
b_null_id_list = b_null_df["id"].tolist()
submission_df.loc[submission_df['id'].isin(b_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.88, 0.04, 0.08]

same_a_b_df2 = df2[(df2["response_a"] == df2["response_b"])]
same_a_b_id_list = same_a_b_df2["id"].tolist()
submission_df.loc[submission_df['id'].isin(same_a_b_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.06, 0.06, 0.88]
```

## Summary

**Overview**: Developed and optimized a human preference prediction model for dialogue systems based on the gemma-2-9b-it model, improving the accuracy of predicting user preference responses in the dialogue system.

**Key Techniques**:
- **Data Processing**: Utilized 88k official and deduplicated data, performed 20-fold cross-validation (trained on one fold only), and created pseudo-labels for ultrafeedback data, expanding the dataset to over 100,000 entries.
- **Model Optimization**: Fine-tuned the **Gemma2ForSequenceClassification** model using LoRA and performed a three-class classification task. Unique prompt design improved handling of long conversation truncations.
- **Inference and Post-Processing**: Implemented a TTA strategy to improve inference results and applied specific post-processing


## Author and Contact

Daoyuan Li - [Kaggle Profile](https://www.kaggle.com/distiller)

For any questions, please contact Daoyuan Li at lidaoyuan2816@gmail.com.
