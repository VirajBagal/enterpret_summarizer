################################################################################
# File: predict.py                                                             #
# Project: Spindle                                                             #
# Created Date: Saturday, 17th June 2023 4:52:13 pm                            #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Saturday, 17th June 2023 5:19:23 pm                           #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################

import datasets
from random import randrange
import os
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import utils
from functools import partial
import wandb
import argparse
import numpy as np


def main(args):
    class config:
        DATASET_PATH = args.dataset_path
        TEXT_COL_NAME = "Text"
        SUMMARY_COL_NAME = "Summary"
        EXPERIMENT_NAME = args.project_name
        RUN_NAME = args.run_name
        OUTPUT_DIR = os.path.join(args.output_dir, EXPERIMENT_NAME)
        BATCH_SIZE = args.batch_size
        # we want to ignore tokenizer pad token in the loss
        LABEL_PAD_TOKEN_ID = -100
        MODEL_DIR = args.model_dir

    if args.log:
        wandb.login()
        os.environ["WANDB_PROJECT"] = config.EXPERIMENT_NAME

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    dataset = datasets.load_from_disk(config.DATASET_PATH)

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")

    sample = dataset["train"][randrange(len(dataset["train"]))]
    print(f"text: \n{sample[config.TEXT_COL_NAME]}\n---------------")
    print(f"summary: \n{sample[config.SUMMARY_COL_NAME]}\n---------------")

    tokenizer, model = utils.load_model_tokenizer_for_inference(config.MODEL_DIR, device="auto")

    max_source_length, max_target_length = utils.get_max_text_lengths(tokenizer, dataset, config)
    # set these values in config
    config.MAX_SOURCE_LENGTH = max_source_length
    config.MAX_TARGET_LENGTH = max_target_length

    column_names = dataset["test"].column_names
    print("Columns names: ", column_names)
    tokenized_dataset = dataset.map(
        utils.tokenize, batched=True, remove_columns=column_names, fn_kwargs={"tokenizer": tokenizer, "config": config}
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['test'].features)}")

    print("Input ID: ", tokenized_dataset["test"]["input_ids"][0])
    print("Attention Mask: ", tokenized_dataset["test"]["attention_mask"][0])
    print("Label: ", tokenized_dataset["test"]["labels"][0])
    # Metric
    metric = evaluate.load("rouge")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=config.LABEL_PAD_TOKEN_ID, pad_to_multiple_of=8
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        predict_with_generate=True,
        fp16=True,
        # logging & evaluation strategies
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        logging_strategy="steps",
        # how often to log
        logging_steps=10,
        evaluation_strategy="steps",
        # how often to evaluate the model
        eval_steps=100,
        save_strategy="steps",
        # how often to checkpoint the model
        save_steps=500,
        report_to="wandb" if args.log else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
        run_name=config.RUN_NAME,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=partial(utils.compute_metrics, tokenizer=tokenizer, metric=metric, config=config),
    )

    # Start training
    prediction_results = trainer.predict(tokenized_dataset["test"])
    if args.log:
        metrics = prediction_results.metrics
        trainer.log_metrics("predict", metrics)

    predictions = prediction_results.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    predictions = [pred.strip() for pred in predictions]
    assert len(predictions) == len(tokenized_dataset["test"])
    tokenized_dataset["test"] = tokenized_dataset["test"].map(
        lambda row, index: {"prediction": predictions[index]}, with_index=True
    )
    tokenized_dataset["test"].to_csv(config.RUN_NAME + ".csv")
    # output_prediction_file = os.path.join(config.OUTPUT_DIR, "generated_predictions.txt")
    # with open(output_prediction_file, "w") as writer:
    #     writer.write("\n".join(predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="path to data folder")
    parser.add_argument("--output_dir", default="../output", help="path to save checkpoints")
    parser.add_argument("--project_name", default="FeedbackSummarizer", help="name of the project")
    parser.add_argument("--run_name", required=True, help="name of the experiment")
    parser.add_argument("--model_dir", required=True, help="directory to saved model checkpoints")
    parser.add_argument("--batch_size", default=2, type=int, help="training and eval batch size")
    parser.add_argument("--log", action="store_true", help="log results to wandb")

    args = parser.parse_args()
    main(args)
