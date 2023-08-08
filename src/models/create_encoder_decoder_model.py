import argparse

from transformers import EncoderDecoderModel, AutoTokenizer


def main(data_args):
    tokenizer = AutoTokenizer.from_pretrained(data_args.checkpoint)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(data_args.checkpoint, data_args.checkpoint)

    # Set special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    # Model Configs from BART
    # See https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=t004XKklvU6q
    model.config.max_length = data_args.max_length
    model.config.min_length = data_args.min_length
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    model.save_pretrained(data_args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="EncoderDecoder", description="Creates EncoderDecoder Model from checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="bert-base-multilingual-cased",
                        help="Model Checkpoint to create EncoderDecoderModel from. Defaults to bert-base-uncased.")
    parser.add_argument("--output", type=str, default="../../models/bert2bert-base-multilingual-cased",
                        help="Output dir to save model")
    parser.add_argument("--max_length", type=int, default=146, help="Max Length of model")
    parser.add_argument("--min_length", type=int, default=5, help="Min Length of model")

    main(data_args=parser.parse_args())
