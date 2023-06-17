################################################################################
# File: utils.py                                                               #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 8:50:10 am                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Saturday, 17th June 2023 4:56:42 pm                           #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel, PeftConfig
import datasets

nltk.download("punkt")


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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model_tokenizer(model_id, use_peft):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load model from the hub
    print("Loading the model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

    print_trainable_parameters(model)

    if use_peft:
        # Define LoRA Config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            # target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
        )
        # prepare int-8 model for training
        # print("Preparing model for int8 training")
        # model = prepare_model_for_int8_training(model)

        # without below code, none of the tensors have grad_fn
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return tokenizer, model


def load_model_tokenizer_for_inference(model_dir, device):
    config = PeftConfig.from_pretrained(model_dir)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model. Offload folder needed to keep offload some weights to disk if RAM is full.
    model = PeftModel.from_pretrained(model, model_dir, device_map=device, offload_folder=model_dir)
    model.eval()

    print("Peft model loaded")
    return model, tokenizer


def get_max_text_lengths(tokenizer, dataset, config):
    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = datasets.concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x[config.TEXT_COL_NAME], truncation=True),
        batched=True,
        remove_columns=[config.TEXT_COL_NAME, config.SUMMARY_COL_NAME],
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = datasets.concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x[config.SUMMARY_COL_NAME], truncation=True),
        batched=True,
        remove_columns=[config.TEXT_COL_NAME, config.SUMMARY_COL_NAME],
    )
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")
    return max_source_length, max_target_length


def tokenize(sample, tokenizer, config, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample[config.TEXT_COL_NAME]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=config.MAX_SOURCE_LENGTH, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample[config.SUMMARY_COL_NAME],
        max_length=config.MAX_TARGET_LENGTH,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 (LABEL_PAD_TOKEN_ID) when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else config.LABEL_PAD_TOKEN_ID) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, metric, config):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 (LABEL_PAD_TOKEN_ID) in the labels as we can't decode them.
    labels = np.where(labels != config.LABEL_PAD_TOKEN_ID, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result
