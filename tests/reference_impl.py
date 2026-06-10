"""Verbatim copy of the competition tokenization, used as a golden reference.

This is `tokenize_cls_p3` exactly as it appears in `competition/1.train.py`
(the code that produced the 4th-place submission). Do not modernize or
refactor this file: its purpose is to pin the library's default behavior to
the competition behavior, byte for byte.
"""


def tokenize_cls_p3(example, tokenizer, max_length):
    input_ids = []
    attention_mask = []
    dot_tokens = tokenizer("......", add_special_tokens=False)["input_ids"]
    final_p_tokens = tokenizer("\n\n---\nWhich response is better? [A or B or tie]\nAnswer: ", add_special_tokens=False)["input_ids"]
    for ps, ras, rbs in zip(example['prompt'], example['response_a'], example['response_b']):
        one_input_ids = [tokenizer.bos_token_id] # All tokens of one sample
        prev_tokens_num = 2 + len(final_p_tokens) # 2 for bos_token and eos_token
        for idx, (p, ra, rb) in enumerate(zip(ps, ras, rbs)):
            r_tokens  = tokenizer(f'\n\n## Round {idx+1}:' if idx else f'## Round {idx+1}:', add_special_tokens=False)["input_ids"] # For Round 1, no leading newline is needed
            p_tokens  = tokenizer(f'\n### Prompt:\n{p}', add_special_tokens=False)["input_ids"]
            ra_tokens = tokenizer(f'\n\n### Response A:\n{ra}', add_special_tokens=False)["input_ids"]
            rb_tokens = tokenizer(f'\n\n### Response B:\n{rb}', add_special_tokens=False)["input_ids"]
            all_tokens_num = prev_tokens_num + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens)

            # If adding the current round's tokens would exceed max_length
            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num - len(r_tokens) - 3*len(dot_tokens)  # Remaining number of p, a, b tokens that can be allocated
                if remain_tokens_num >= 80:
                    # If the allocatable p, a, b token count is > 80, the p, a, b tokens can be truncated
                    p_tokens  =  p_tokens[:int(remain_tokens_num*0.2)] + dot_tokens if len( p_tokens) > int(remain_tokens_num*0.2) else  p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
                    one_input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens # Add to input_ids
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
