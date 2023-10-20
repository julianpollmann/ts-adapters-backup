import argparse

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    EarlyStoppingCallback
)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]
LANGUAGE_MAPPING = {
    "en": "en_XX",
    "es": "es_XX",
    "fa": "fa_IR",
    "fr": "fr_XX"
}


def main(data_args: argparse.Namespace, ds_train=None, ds_valid=None):
    dataset = load_dataset(data_args.dataset, name=data_args.language)

    tokenizer = AutoTokenizer.from_pretrained(data_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(data_args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
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

    # Define LoRA Config
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q", "v"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.SEQ_2_SEQ_LM
    # )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules=["q_proj", "v_proj"],
        r=data_args.rank,
        lora_alpha=data_args.lora_alpha,
        lora_dropout=0.1
    )

    # prepare int-8 model for training
    # model = prepare_model_for_int8_training(model)

    # add adapter
    model = get_peft_model(model, peft_config)

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
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    train_result = trainer.train()
    model.save_pretrained(data_args.output_dir)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    metrics["total_flos"] = trainer.state.total_flos
    metrics["model_params"] = all_params
    metrics["trainable_params"] = trainable_params
    metrics["trainable_params_share"] = round((trainable_params / all_params) * 100, 2)

    trainer.log_metrics(split="train", metrics=metrics)
    trainer.save_metrics(split="train", metrics=metrics)
    trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Text Simplification Finetuning",
                                     description="Finetuning Text Simplification")
    parser.add_argument("output_dir", type=str, help="Output dir to store trained checkpoints")
    parser.add_argument("dataset", type=str, help="Name of the dataset in Huggingface Datasets format")
    parser.add_argument("--checkpoint", type=str, default="facebook/mbart-large-50",
                        help="Model Checkpoint, can be either a pretrained model or a own checkpoint. Defaults to mbart-large-50.")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps to train")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input length of tokenized text. Defaults to 1024")
    parser.add_argument("--max_target_length", type=int, default=1024,
                        help="Max target length of tokenized text. Defaults to 1024")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)

    main(data_args=parser.parse_args())
