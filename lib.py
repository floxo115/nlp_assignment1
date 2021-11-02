import re
from collections import Counter
from pathlib import Path
from string import punctuation
from typing import Tuple

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def import_datasets(size: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param size: "medium"|"small" selects the loaded datasets
    :return: train / test / validation sets as Dataframes
    """

    path = Path(".", "datasets")

    train_path = path.joinpath(Path(f"thedeep.{size}.train.txt"))
    test_path = path.joinpath(Path(f"thedeep.{size}.test.txt"))
    validation_path = path.joinpath(Path(f"thedeep.{size}.validation.txt"))

    names = ["id", "text", "label"]
    index_col = "id"
    train_df: pd.DataFrame = pd.read_csv(train_path, names=names, index_col=index_col)
    test_df: pd.DataFrame = pd.read_csv(test_path, names=names, index_col=index_col)
    validation_df: pd.DataFrame = pd.read_csv(validation_path, names=names, index_col=index_col)

    return train_df, test_df, validation_df


def preprocess_txt(input_df: pd.DataFrame):
    """
    Takes a DataFrame representation of the text datasets and preprocesses it.
    :param input_df: DataFrame representing a dataset
    :return: None
    """

    def remove_punctuation(text):
        text = text.translate(str.maketrans('', '', punctuation))
        return text

    def remove_stopwords(text):
        text = [word for word in text.split(" ") if word not in stopwords.words("english")]
        return " ".join(text)

    def remove_numbers(text):
        text = re.sub(r"\d+", "", text)
        return text

    def stemming(text):
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word) for word in text.split(" ")])
        return text

    processing_steps = [str.lower, remove_stopwords, remove_numbers, stemming]
    for step in processing_steps:
        input_df["text"] = input_df["text"].apply(step)


def get_n_most_common_tokens(input_df, n):
    counter = Counter(" ".join(input_df["text"].values).split(" "))
    del counter[""]
    return counter.most_common(n)


if __name__ == '__main__':
    pass
    # train_df, test_df, val_df = import_datasets("small")
    # print(get_n_most_common_tokens(train_df, 20))
    # preprocess_txt(train_df)
    # print(get_n_most_common_tokens(train_df, 20))
    # pprint(train_df)
