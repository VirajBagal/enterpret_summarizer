################################################################################
# File: lambda_handler.py                                                      #
# Project: Spindle                                                             #
# Created Date: Sunday, 25th June 2023 10:38:47 pm                             #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Sunday, 25th June 2023 10:49:01 pm                            #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import json
import inference_utils

max_target_length = 432  # max tokens in whole dataset summary
device = "auto"
weight_path = inference_utils.download_checkpoint()
print("Path of downloaded artifacts: ", weight_path)
# Load peft config for pre-trained checkpoint etc.
model, tokenizer = inference_utils.load_model_tokenizer(weight_path, device)


def summarize_text(text):
    processed_text = inference_utils.preprocess_text(text)
    # we had trained in this way for Flan-T5
    processed_text = "summarize: " + processed_text
    input_ids = tokenizer(processed_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_target_length, do_sample=False)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    summary = None if summary.strip() == "Nothing" else summary
    return summary


def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence
    """

    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Got the input: {body['text']}")
        response = summarize_text(body["text"])
        return {"statusCode": 200, "headers": {}, "body": json.dumps(response)}
    else:
        return summarize_text(event["text"])
