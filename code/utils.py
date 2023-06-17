################################################################################
# File: utils.py                                                               #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 8:50:10 am                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Friday, 16th June 2023 6:49:56 pm                             #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import re
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")


# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def preprocess_text(text):
    # Remove strict links
    text = re.sub(r"https?:\/\/\S+", "", text)

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    text = emoji_pattern.sub("", text)

    # Remove any remaining special characters or punctuations
    text = re.sub(r"[^\w\s]", "", text)

    text = text.replace("STRICT_LINK", "")

    return text
