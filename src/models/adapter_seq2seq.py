import argparse

from os import path
from datasets import load_dataset
from transformers import (
    Seq2SeqAdapterTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast, EarlyStoppingCallback
)
from transformers.adapters import AdapterConfig

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]
LANGUAGE_MAPPING = {
    "en": "en_XX",
    "es": "es_XX",
    "fa": "fa_IR",
    "fr": "fr_XX"
}

def get_text_len(dataset, text, tokenizer):
    tokenized_targets_train = dataset["train"].map(
        lambda x: tokenizer(x[text], truncation=True),
        batched=True,
        remove_columns=["src", "tgt"],
    )
    tokenized_targets_valid = dataset["train"].map(
        lambda x: tokenizer(x[text], truncation=True),
        batched=True,
        remove_columns=["src", "tgt"],
    )
    train_target_length = [len(x) for x in tokenized_targets_train["input_ids"]]
    valid_target_length = [len(x) for x in tokenized_targets_valid["input_ids"]]

    return max(train_target_length + valid_target_length)


def main(data_args: argparse.Namespace):
    dataset = load_dataset(data_args.dataset, name=data_args.language)

    config = AutoConfig.from_pretrained(data_args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(data_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(data_args.checkpoint, config=config)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        lang = LANGUAGE_MAPPING.get(data_args.language) if data_args.language is not None else LANGUAGE_MAPPING.get("en")

        tokenizer.src_lang = lang
        tokenizer.tgt_lang = lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        # forced_bos_token_id = (
        #     tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        # )
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[lang]

    max_source_length = get_text_len(dataset, "src", tokenizer)
    max_target_length = get_text_len(dataset, "tgt", tokenizer)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['src'],
            max_length=max_source_length,
            truncation=True,
            padding=True
        )
        outputs = tokenizer(
            text_target=examples["tgt"],
            max_length=max_target_length,
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

    # Add new adapter
    adapter_name = "simplification"
    adapter_config = AdapterConfig.load(
        config=data_args.adapter_config,
        reduction_factor=data_args.reduction_factor,
    )
    model.add_adapter(adapter_name, config=adapter_config)

    # Activate for training
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        remove_unused_columns=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=data_args.batch_size,
        per_device_eval_batch_size=data_args.batch_size,
        learning_rate=data_args.learning_rate,
        max_steps=data_args.steps,
        save_total_limit=5,
        optim="adamw_torch",
        fp16=True,
        eval_steps=100,
        logging_steps=100,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqAdapterTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    metrics["total_flos"] = trainer.state.total_flos

    trainer.log_metrics(split="train", metrics=metrics)
    trainer.save_metrics(split="train", metrics=metrics)
    trainer.save_state()
    model.save_adapter(path.join(data_args.output_dir, adapter_name), adapter_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Text Simplification Finetuning',
                                     description='Finetuning Text Simplification')
    parser.add_argument('output_dir', type=str, help='Output dir to store trained checkpoints')
    parser.add_argument('dataset', type=str, help='Name of the dataset in Huggingface Datasets format')
    parser.add_argument("--checkpoint", type=str, default="facebook/mbart-large-50",
                        help='Model Checkpoint, can be either a pretrained model or a own checkpoint.')
    parser.add_argument('--steps', type=int, default=400, help='Number of steps to train')
    parser.add_argument('--max_input_length', type=int, default=1024,
                        help='Max input length of tokenized text. Defaults to 1024')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--learning_rate", type=str, default=2e-5)
    parser.add_argument("--adapter_config", type=str, default="pfeiffer")
    parser.add_argument("--language", type=str, default=None)

    main(data_args=parser.parse_args())
