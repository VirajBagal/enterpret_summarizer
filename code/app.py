################################################################################
# File: app.py                                                                 #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 7:06:42 pm                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Saturday, 17th June 2023 12:01:44 am                          #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################

import inference_utils
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class TextRequest(BaseModel):
    text: str


text = "I am not able to install zoom"
max_target_length = 432  # max tokens in whole dataset summary

# Load peft config for pre-trained checkpoint etc.
peft_model_dir = "lora_weights"
device = "auto"
model, tokenizer = inference_utils.load_model_tokenizer(peft_model_dir, device)


@app.post("/summarize")
def summarize_text(request: TextRequest):
    processed_text = inference_utils.preprocess_text(text)
    input_ids = tokenizer(processed_text, return_tensors="pt", truncation=True).input_ids
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=False)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return {"summary": summary}
