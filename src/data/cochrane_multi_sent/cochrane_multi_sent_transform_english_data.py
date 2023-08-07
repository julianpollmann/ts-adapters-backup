import pandas as pd
from os import path, makedirs


def main():
    file_path = "/homes/julez/ts-adapters/src/data/cochrane_multi_sent/en"
    if not path.exists(file_path):
        makedirs(file_path)

    for split in ["train", "test", "val"]:
        df = pd.read_csv(f"/homes/julez/datasets-raw/cochrane_sent/unfiltered (r=0)/en/{split}0_en.csv",
                         index_col=0)
        df.rename(columns={"input_text": "src", "target_text": "tgt"}, inplace=True)
        df.drop(["prefix"], axis=1, inplace=True)
        df["lang"] = "en"
        df["split"] = split

        if split == "val":
            df.to_csv(f"{file_path}/validation_aligned.csv", index=False)
        else:
            df.to_csv(f"{file_path}/{split}_aligned.csv", index=False)


if __name__ == "__main__":
    main()
