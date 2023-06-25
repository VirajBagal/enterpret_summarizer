################################################################################
# File: app_v2.py                                                              #
# Project: Spindle                                                             #
# Created Date: Sunday, 25th June 2023 2:53:46 pm                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Sunday, 25th June 2023 4:00:01 pm                             #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################

import inference_utils
from fastapi import FastAPI
from pydantic import BaseModel
import logging

logging.basicConfig(filename="requests.log", encoding="utf-8", level=logging.DEBUG)
logger = logging.getLogger("root")

app = FastAPI()


class SummaryRequest(BaseModel):
    text: str
    product_name: str
    record_type: str


max_target_length = 432  # max tokens in whole dataset summary
device = "auto"
weight_path = inference_utils.download_checkpoint()
print("Path of downloaded artifacts: ", weight_path)
# Load peft config for pre-trained checkpoint etc.
model, tokenizer = inference_utils.load_model_tokenizer(weight_path, device)


@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    logger.info(f"Product Name: {request.product_name}")
    logger.info(f"Record Type: {request.record_type}")
    logger.info(f"Text: {request.text}")
    processed_text = inference_utils.preprocess_text(request.text)
    # we had trained in this way for Flan-T5
    processed_text = "summarize: " + processed_text
    input_ids = tokenizer(processed_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=False)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    summary = None if summary.strip() == "Nothing" else summary
    logger.info(f"Summary: {summary}")
    return {"summary": summary}
