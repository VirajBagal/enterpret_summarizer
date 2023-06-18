################################################################################
# File: app.py                                                                 #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 7:06:42 pm                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Monday, 19th June 2023 12:18:11 am                            #
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
    processed_text = inference_utils.preprocess_text(request.text)
    # we had trained in this way for Flan-T5
    processed_text = "summarize: " + processed_text
    input_ids = tokenizer(processed_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=False)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return {"summary": summary}
