import argparse

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def main(data_args: argparse.Namespace):

    train_data, valid_data = [], []
    if data_args.training_languages:
        for language in data_args.training_languages:
            train_data.append(load_dataset(data_args.dataset, name=language, split="train"))
            valid_data.append(load_dataset(data_args.dataset, name=language, split="validation"))

    if data_args.few_shot_language:
        train_data.append(load_dataset(data_args.dataset, name=data_args.few_shot_language, split="train[:10%]"))
        valid_data.append(load_dataset(data_args.dataset, name=data_args.few_shot_language, split="validation"))

    dataset = DatasetDict({"train": concatenate_datasets(train_data), "validation": concatenate_datasets(train_data)})

    tokenizer = AutoTokenizer.from_pretrained(data_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(data_args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["src"],
            max_length=data_args.max_input_length,
            truncation=True,
            padding=True
        )
        labels = tokenizer(
            examples["tgt"],
            max_length=data_args.max_target_length,
            truncation=True,
            padding=True
        )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    column_names = dataset["train"].column_names
    dataset = dataset.map(preprocess_function, remove_columns=column_names, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        remove_unused_columns=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=data_args.batch_size,
        per_device_eval_batch_size=data_args.batch_size,
        learning_rate=data_args.learning_rate,
        max_steps=data_args.steps,
        save_total_limit=3,
        optim="adamw_torch",
        #predict_with_generate=True,
        #include_inputs_for_metrics=True,
        fp16=True,
        eval_steps=1000,
        logging_steps=1000,
        weight_decay=0.01,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    metrics["total_flos"] = trainer.state.total_flos
    metrics["model_params"] = model.num_parameters()

    trainer.log_metrics(split="train", metrics=metrics)
    trainer.save_metrics(split="train", metrics=metrics)
    trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Text Simplification Finetuning",
                                     description="Finetuning Text Simplification")
    parser.add_argument("output_dir", type=str, help="Output dir to store trained checkpoints")
    parser.add_argument("dataset", type=str, help="Name of the dataset in Huggingface Datasets format")
    parser.add_argument("--checkpoint", type=str, default="facebook/bart-large",
                        help="Model Checkpoint, can be either a pretrained model or a own checkpoint. Defaults to bart-base.")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps to train")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input length of tokenized text. Defaults to 1024")
    parser.add_argument("--max_target_length", type=int, default=1024,
                        help="Max target length of tokenized text. Defaults to 1024")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--training_languages", type=str, nargs='+')
    parser.add_argument("--few_shot_language", type=str, default=None)
    parser.add_argument("--learning_rate", type=str, default=2e-5)

    main(data_args=parser.parse_args())
