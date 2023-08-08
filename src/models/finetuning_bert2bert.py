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


def main(data_args: argparse.Namespace):
    dataset = load_dataset(data_args.dataset, name=data_args.language)

    # Load Tokenizer, Model and DataCollator
    # EncoderDecoderModel is warm-started with pre-trained BERT models
    tokenizer = AutoTokenizer.from_pretrained(data_args.base_model)
    model = EncoderDecoderModel.from_pretrained(data_args.base_encdec_model)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
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
        outputs = tokenizer(
            examples["tgt"],
            max_length=data_args.max_target_length,
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

    training_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        remove_unused_columns=False,
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
        eval_steps=500,
        logging_steps=500,
        weight_decay=0.01,
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
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Max input length of tokenized text. Defaults to 512")
    parser.add_argument("--max_target_length", type=int, default=512,
                        help="Max target length of tokenized text. Defaults to 512")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--learning_rate", type=str, default=1e-4)
    parser.add_argument("--language", type=str, default=None)

    main(data_args=parser.parse_args())
