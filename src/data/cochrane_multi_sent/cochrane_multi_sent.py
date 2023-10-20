import csv
import os
import pandas as pd
from typing import List

import datasets


class CochranePara(datasets.GeneratorBasedBuilder):
    """Dataset Loader for Cochrane DS for Text Simplification on Sent Level."""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="en", version=VERSION, description="English"),
        datasets.BuilderConfig(name="es", version=VERSION, description="Spanish"),
        datasets.BuilderConfig(name="fr", version=VERSION, description="French"),
        datasets.BuilderConfig(name="fa", version=VERSION, description="Farsi"),
    ]
    DEFAULT_CONFIG_NAME = "en"

    def _info(self):
        features = datasets.Features(
            {
                "doi": datasets.Value("string"),
                "src": datasets.Value("string"),
                "tgt": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description="Cochrane Simplifications on Sents Level",
            features=features,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        generator = []
        file_train = dl_manager.download(os.path.join(self.config.name, "train_filtered.csv"))
        file_test = dl_manager.download(os.path.join(self.config.name, "test.csv"))
        file_eval = dl_manager.download(os.path.join(self.config.name, "valid.csv"))

        if os.path.isfile(file_train):
            train = datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_train,
                    "split": "train",
                },
            )
            generator.append(train)

        if os.path.isfile(file_test):
            test = datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": file_test,
                    "split": "test",
                },
            )
            generator.append(test)

        if os.path.isfile(file_eval):
            valid = datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": file_eval,
                    "split": "eval",
                },
            )
            generator.append(valid)

        return generator

    def _generate_examples(self, filepath, split):
        df = pd.read_csv(filepath, index_col=0)

        for index, row in df.iterrows():
            yield index, {
                "doi": row["doi"],
                "src": row["src"],
                "tgt": row["tgt"]
            }