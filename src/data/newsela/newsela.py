import pandas as pd

from os import path, getcwd
from typing import List

import datasets


class Newsela(datasets.GeneratorBasedBuilder):
    """Dataset Loader for Newsela DS for Text Simplification on Sent Level."""

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "src": datasets.Value("string"),
                "tgt": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description="Newsela",
            features=features,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        currentpath = path.join(getcwd(), "ts-adapters", "src", "data", "newsela")
        generator = []

        train = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "filepath": f"{currentpath}/train.csv",
                "split": "train",
            },
        )
        generator.append(train)

        test = datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "filepath": f"{currentpath}/test.csv",
                "split": "test",
            },
        )
        generator.append(test)

        valid = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                "filepath": f"{currentpath}/valid.csv",
                "split": "eval",
            },
        )
        generator.append(valid)

        return generator

    def _generate_examples(self, filepath, split):
        df = pd.read_csv(filepath, index_col=0)

        for index, row in df.iterrows():
            yield index, {
                "src": row["complex_sent"],
                "tgt": row["simple_sent"]
            }
