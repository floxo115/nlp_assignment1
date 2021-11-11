from collections import Counter
from pathlib import Path
from string import punctuation
from typing import Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy.linalg import svd


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

    # def remove_punctuation(text):
    #     text = text.translate(str.maketrans('', '', punctuation))
    #     return text
    #
    # def remove_stopwords(text):
    #     text = [word for word in text.split(" ") if word not in stopwords.words("english")]
    #     return " ".join(text)
    #
    # def remove_numbers(text):
    #     text = re.sub(r"\d+", "", text)
    #     return text
    #
    # def stemming(text):
    #     stemmer = PorterStemmer()
    #     text = " ".join([stemmer.stem(word) for word in text.split(" ")])
    #     return text
    #
    def preprocess_txt(text: str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', punctuation))
        words_lst = text.split(" ")

        # stem words with PorterStemmer
        stemmer = PorterStemmer()

        for i in range(len(words_lst)):

            if words_lst[i] in stopwords.words("english") or words_lst[i].isnumeric():
                words_lst[i] = ""

            words_lst[i] = stemmer.stem(words_lst[i])

        return " ".join([word for word in words_lst if word != ""])

    # processing_steps = [str.lower, remove_punctuation, remove_stopwords, remove_numbers, stemming]

    input_df["text"] = input_df["text"].apply(preprocess_txt)


def get_n_most_common_tokens(input_df, n):
    counter = Counter(" ".join(input_df["text"].values).split(" "))
    del counter[""]
    return counter.most_common(n)


def reduce_words(input_df: pd.DataFrame, full_dict: pd.DataFrame, threshold: int):
    """removing words that are below the threshold from the corpus"""

    def remove_words(word_list):
        def inner(text):
            text = " ".join([w for w in text.split(" ") if w not in word_list])
            return text

        return inner

    word_list = (full_dict[full_dict["count"] < threshold])["token"].to_list()
    input_df["text"] = input_df["text"].apply(remove_words(set(word_list)))


def create_tc_vec_from_doc(input_df, n):
    word_tokens = input_df.iloc[n]["text"]
    tokens = word_tokenize(word_tokens)
    fd = FreqDist(tokens)

    doc_vec = np.zeros(len(corpus_dict))

    tokens = corpus_dict["token"]
    for i, token in enumerate(tokens):
        doc_vec[i] = fd[token]

    return doc_vec


def create_tc_vec_from_doc(input_df, corpus_dict, n):
    word_tokens = input_df.iloc[n]["text"]
    tokens = word_tokenize(word_tokens)
    fd = FreqDist(tokens)

    doc_vec = np.zeros(len(corpus_dict))

    tokens = corpus_dict["token"]
    for i, token in enumerate(tokens):
        doc_vec[i] = fd[token]

    return doc_vec


def latent_semantic_analysis(arr: np.ndarray, k) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, Sigma, V_tp = svd(arr)
    Sigma = np.diag(Sigma)
    U_k = U[:, :k]
    Sigma_k = Sigma[:k, :k]
    V_tp_k = V_tp[:k, :]

    return U_k, Sigma_k, V_tp_k

def compute_sparsity(arr: np.ndarray) -> int:
    empty = (arr == 0).sum()
    return  empty / arr.size

if __name__ == '__main__':
    from pprint import pprint

    train_df, test_df, val_df = import_datasets("small")
    train_df = train_df.iloc[range(0,10)]
    print(get_n_most_common_tokens(train_df, 20))
    preprocess_txt(train_df)
    print(get_n_most_common_tokens(train_df, 20))

    token_counter = Counter((" ".join(train_df["text"].values).split(" ")))

    corpus_dict = pd.DataFrame({"token": token_counter.keys(), "count": token_counter.values()})
    corpus_dict = corpus_dict.sample(frac=1).reset_index(drop=True)

    threshold = 3

    reduced_corpus_dict = corpus_dict[corpus_dict["count"] >= threshold]

    pprint(train_df["text"])
    reduce_words(train_df, corpus_dict, threshold)
    pprint(train_df["text"])

    train_tc_vecs = []
    train_tf_vecs = []
    for doc_idx in range(len(train_df)):
        train_tc_vecs.append(create_tc_vec_from_doc(train_df, corpus_dict, doc_idx))
        train_tf_vecs.append(np.log(train_tc_vecs[-1] + 1))

    train_tc_vecs = np.array(train_tc_vecs).T
    train_tf_vecs = np.array(train_tf_vecs).T

    pprint(train_tc_vecs)
    pprint(train_tf_vecs)

    print(f"sparsity of training vector: {compute_sparsity(train_tc_vecs)}")

    k = 5
    lsa_train_tc = latent_semantic_analysis(train_tc_vecs, k)
    lsa_train_tf = latent_semantic_analysis(train_tf_vecs, k)
