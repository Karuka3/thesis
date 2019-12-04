import pandas as pd
import numpy as np
import chardet
import codecs
import os
import random
from textblob import TextBlob
from RandomWord import RandomWords
from TopicModel import Preprocessing


def main():
    path = r"C:\Users\Kazuki\thesis\data"
    os.chdir(path)
    """
    file = "Tweets.csv"
    data = pd.read_csv(file)
    texts = data["text"]
    """
    r = RandomWords()
    randomcount = [random.randint(1, 150) for i in range(10000)]
    docs = []
    for k in randomcount:
        word = r.get_random_words(limit=k)
        randomtext = [" ".join(word)]
        docs.append(randomtext)


if __name__ == "__main__":
    main()
