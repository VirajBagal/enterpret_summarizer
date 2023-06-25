################################################################################
# File: eval_utils.py                                                          #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 6:49:50 pm                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Sunday, 25th June 2023 3:49:22 pm                             #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################

import re
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import wandb


def load_model_tokenizer(model_dir, device):
    config = PeftConfig.from_pretrained(model_dir)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model. Offload folder needed to keep offload some weights to disk if RAM is full.
    model = PeftModel.from_pretrained(model, model_dir, device_map=device, offload_folder=model_dir)
    model.eval()

    print("Peft model loaded")
    return model, tokenizer


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


def download_checkpoint():
    run = wandb.init()
    artifact = run.use_artifact("vbagal/model-registry/review_summarizer:production", type="model")
    artifact_dir = artifact.download()
    return artifact_dir
