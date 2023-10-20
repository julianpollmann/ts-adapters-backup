import argparse
import logging

from datasets import load_dataset
# from easse.fkgl import corpus_fkgl
# from easse.sari import corpus_sari
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EncoderDecoderModel
)

logger = logging.getLogger(__name__)


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

    # Load Tokenizer, Model and DataCollator
    # EncoderDecoderModel is warm-started with pre-trained BERT models
    tokenizer = AutoTokenizer.from_pretrained(data_args.base_model)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    max_source_length = get_text_len(dataset, "src", tokenizer)
    max_target_length = get_text_len(dataset, "tgt", tokenizer)

    def preprocess_function(examples, padding="max_length"):
        model_inputs = tokenizer(
            examples["src"],
            max_length=max_source_length,
            padding=padding,
            truncation=True
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=max_target_length,
            padding=padding,
            truncation=True
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
    dataset = dataset.map(
        preprocess_function,
        remove_columns=column_names,
        batched=True,
        batch_size=data_args.batch_size
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        data_args.base_model,
        data_args.base_model,
        tie_encoder_decoder=True
    )

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    bad_words = ['[CLS]']
    bad_words_ids = [tokenizer.vocab[token] for token in bad_words]
    model.config.bad_words_ids = [bad_words_ids]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

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
        # predict_with_generate=True,
        # include_inputs_for_metrics=True,
        fp16=True,
        eval_steps=500,
        logging_steps=500,
        # weight_decay=0.01,
        warmup_ratio=0.1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
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
    parser = argparse.ArgumentParser(prog="TSFinetuningBert2Bert",
                                     description="Text Simplification with Finetuning and EncoderDecoder Models.")
    parser.add_argument("output_dir", type=str, help="Output dir to store trained checkpoints")
    parser.add_argument("dataset", type=str, help="Name of the dataset in Huggingface Datasets format")
    parser.add_argument("--base_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--base_encdec_model", type=str, default="../../models/bert2bert-base-multilingual-cased")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--language", type=str, default=None)

    main(data_args=parser.parse_args())
