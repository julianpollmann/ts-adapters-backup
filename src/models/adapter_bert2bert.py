import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AdapterConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    EncoderDecoderModel)
from transformers.adapters import Seq2SeqAdapterTrainer
from transformers.adapters.composition import Stack
from os import path


def main(args):
    dataset = load_dataset(args.dataset, name=args.lang_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = EncoderDecoderModel.from_pretrained(args.base_encdec_model)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["src"],
            max_length=512,
            truncation=True,
            padding=True
        )
        outputs = tokenizer(
            examples["tgt"],
            max_length=512,
            truncation=True,
            padding=True
        )
        model_inputs["labels"] = outputs["input_ids"]
        model_inputs["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                                  outputs.input_ids]
        return model_inputs

    column_names = dataset["train"].column_names
    dataset = dataset.map(preprocess_function, remove_columns=column_names, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    if args.lang_adapter:
        # Language adapter
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        model.load_adapter(f"{args.lang_adapter}/wiki@ukp", config=lang_adapter_config, model_name=args.base_model)

    # Task adapter
    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=args.reduction_factor)
    model.add_adapter("simplification", config=adapter_config)

    # Activate the adapter
    model.train_adapter("simplification")

    # Stack adapters if language adapter configured
    if args.lang_adapter:
        model.active_adapters = Stack(args.lang_adapter, "simplification")
    else:
        model.set_active_adapters("simplification")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        max_steps=args.steps,
        save_total_limit=3,
        optim="adamw_torch",
        fp16=True,
        weight_decay=0.01,
        eval_steps=500,
        logging_steps=500,
    )

    trainer = Seq2SeqAdapterTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    train_result = trainer.train()

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    metrics["total_flos"] = trainer.state.total_flos

    trainer.log_metrics(split="train", metrics=metrics)
    trainer.save_metrics(split="train", metrics=metrics)
    trainer.save_state()
    model.save_adapter(path.join(args.output_dir, "simplification"), "simplification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TSAdapterBert2Bert",
                                     description="Text Simplification with Adapters and EncoderDecoder Models.")
    parser.add_argument("output_dir", type=str, help="Output dir to store trained checkpoints")
    parser.add_argument("dataset", type=str, help="Name of the dataset in Huggingface Datasets format")
    parser.add_argument("--base_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--base_encdec_model", type=str, default="../../models/bert2bert-base-multilingual-cased")
    parser.add_argument("--lang_dataset", type=str, default=None, help="Language of the dataset e.g., 'es'")
    parser.add_argument("--lang_adapter", type=str, default=None, help="Language of the language adapter e.g., 'es'")
    parser.add_argument("--reduction_factor", type=int, default=8, help="Reduction factor of the adapter.")
    parser.add_argument("--steps", type=int, default=100, help="Steps to train.")

    main(args=parser.parse_args())
