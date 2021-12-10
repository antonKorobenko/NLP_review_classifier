import nltk
import os
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

# create stemmer for stemming
STEMMER = SnowballStemmer("english")


def preproc(text: str) -> str:
    """Function which receives raw text(str), then tokenize it with nltk,
    removes all stopwords and punctuation marks.
    Returns str object."""
    words = list()
    text = text.lower()
    tokenized = nltk.word_tokenize(text)
    for token in tokenized:
        token = STEMMER.stem(token)
        if token not in nltk.corpus.stopwords.words("english") and not all(ch in string.punctuation for ch in token):
            words.append(token)

    return " ".join(word for word in words)


def get_data(type_: str, files_to_read: int) -> tuple:
    """Function for reading data in specific folder wit hspecific datastructure."""
    total_files = files_to_read
    files_to_read = int(files_to_read // 2)
    curr_file_num = 0

    # get_all_negative
    negative_reviews = list()
    iterator = 0
    print(f"Getting {type_} data.")
    for filename in os.listdir(f"./aclImdb/{type_}/neg"):
        with open(f"./aclImdb/{type_}/neg/"+str(filename), "r", encoding="utf-8") as f:
            negative_reviews.append(f.read())

        if iterator == files_to_read:
            break

        if curr_file_num % 100 == 0:
            print(f"Read {curr_file_num} of {total_files} files")
        curr_file_num += 1
        iterator += 1

    # get_all_positive
    positive_reviews = list()
    iterator = 0
    for filename in os.listdir(f"./aclImdb/{type_}/pos"):
        with open(f"./aclImdb/{type_}/pos/"+str(filename), "r", encoding="utf-8") as f:
            positive_reviews.append(f.read())

        if iterator == files_to_read:
            break

        if curr_file_num % 100 == 0:
            print(f"Read {curr_file_num} of {total_files} files")
        curr_file_num += 1
        iterator += 1

    print(f"T{type_[1:]} data successfully loaded.")

    return positive_reviews, negative_reviews


def process_model(vectorizer, train_data, test_data):
    train = vectorizer.fit_transform(train_data["review"])
    test = vectorizer.transform(test_data["review"])

    classifier = LogisticRegression(random_state=0).fit(
        train, train_data["type"])

    prediction = classifier.predict(test)
    return accuracy_score(prediction, test_data["type"])


def main():

    # downloaad required packages
    nltk.download("stopwords")
    nltk.download('punkt')

    # read train data from dataset
    train_num = int(input("Number of reviews to train >> "))
    positive_train, negative_train = get_data("train", train_num)
    # create and concat train dataframes
    df_positive_train = pd.DataFrame(positive_train, columns=["review"])
    df_positive_train["type"] = 1
    df_negative_train = pd.DataFrame(negative_train, columns=["review"])
    df_negative_train["type"] = 0

    # concat train dataframes
    train_reviews = pd.concat(
        (df_positive_train, df_negative_train), axis=0).sample(frac=1.0)
    train_reviews.index = range(0, len(train_reviews))

    # clean train memory
    del positive_train
    del negative_train
    del df_positive_train
    del df_negative_train

    # read test data from dataset
    test_num = int(input("Number of reviews to test >> "))
    positive_test, negative_test = get_data("test", test_num)
    # create and concat test dataframes
    df_positive_test = pd.DataFrame(positive_test, columns=["review"])
    df_positive_test["type"] = 1
    df_negative_test = pd.DataFrame(negative_test, columns=["review"])
    df_negative_test["type"] = 0

    # concat test dataframes
    test_reviews = pd.concat(
        (df_positive_test, df_negative_test), axis=0).sample(frac=1.0)
    test_reviews.index = range(0, len(test_reviews))

    # clean train memory
    del positive_test
    del negative_test
    del df_positive_test
    del df_negative_test

    # tokenize + stem + stopwords (validating\formatting text)
    print("Validating train data.")
    train_reviews["review"] = train_reviews["review"].apply(preproc)
    print("Train data was successfully validated.\n"+"Validating test data")
    test_reviews["review"] = test_reviews["review"].apply(preproc)
    print("Test data was successfully validated.")

    # BOW
    vectorizer_BOW = CountVectorizer()

    # TF-IDF
    vectorizer_TFIDF = TfidfVectorizer()

    # BOW with diagrams
    vectorizer_BOW_bi = CountVectorizer(ngram_range=(1, 3))
    res = []
    for v in (vectorizer_BOW, vectorizer_BOW_bi, vectorizer_TFIDF):
        res.append(process_model(v, train_reviews, test_reviews))

    print(f"DOW accuracy = {res[0]*100}%")
    print(f"DOW with diagrams accuracy = {res[1]*100}%")
    print(f"TF-IDF accuracy = {res[2]*100}%")


if __name__ == "__main__":
    main()
