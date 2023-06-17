################################################################################
# File: generate.py                                                            #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 3:40:29 pm                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Friday, 16th June 2023 7:00:35 pm                             #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################

import inference_utils

text = "I am not able to install zoom"
max_target_length = 432  # max tokens in whole dataset summary

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "../output"
device = "auto"
model, tokenizer = inference_utils.load_model_tokenizer(peft_model_id, device)


def summarize(text):
    processed_text = inference_utils.preprocess_text(text)
    input_ids = tokenizer(processed_text, return_tensors="pt", truncation=True).input_ids
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=False)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return summary


summary = summarize(text)
print(f"input sentence: {text}\n{'---'* 20}")
print(f"summary:\n{summary}")
