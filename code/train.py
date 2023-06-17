################################################################################
# File: train.py                                                               #
# Project: Spindle                                                             #
# Created Date: Friday, 16th June 2023 8:12:09 am                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Saturday, 17th June 2023 10:06:07 am                          #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import datasets
from random import randrange
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


DATASET_PATH = "data_v3"
TEXT_COL_NAME = "Text"
SUMMARY_COL_NAME = "Summary"
EXPERIMENT_NAME = "summary_ann"
OUTPUT_DIR = "../output/{EXPERIMENT_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset = datasets.load_from_disk(DATASET_PATH)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")


sample = dataset["train"][randrange(len(dataset["train"]))]
print(f"text: \n{sample[TEXT_COL_NAME]}\n---------------")
print(f"summary: \n{sample[SUMMARY_COL_NAME]}\n---------------")

model_id = "google/flan-t5-large"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = datasets.concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x[TEXT_COL_NAME], truncation=True),
    batched=True,
    remove_columns=[TEXT_COL_NAME, SUMMARY_COL_NAME],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = datasets.concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x[SUMMARY_COL_NAME], truncation=True),
    batched=True,
    remove_columns=[TEXT_COL_NAME, SUMMARY_COL_NAME],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")


def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample[TEXT_COL_NAME]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample[SUMMARY_COL_NAME], max_length=max_target_length, padding=padding, truncation=True
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


column_names = dataset["train"].column_names
print("Columns names: ", column_names)
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=column_names)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# load model from the hub
print("Loading the model")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
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


BATCH_SIZE = 2
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)


# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    fp16=True,  # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False,
    gradient_checkpointing=True,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
trainer.evaluate()

model.save_pretrained(OUTPUT_DIR)
