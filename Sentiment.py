import pandas as pd
import numpy as np
import chardet
import codecs
import os
import random
import re
from textblob import TextBlob
from RandomWord import RandomWords
from TopicModel import Preprocessing


def text_preprocessing(text):
    no_space = re.compile(r"[.;:!\'?,\"()\[\]]")
    # number = re.compile(r"[0-9]+")
    text = [no_space.sub("", line.lower()) for line in text]
    # text = [number.sub("0", line) for line in text]
    return text


def get_sentiment(docs):
    positive = {"data": [], "polarity": []}
    negative = {"data": [], "polarity": []}
    neutral = {"data": [], "polarity": []}
    for texts in docs:
        clean_text = text_preprocessing(texts)
        text = TextBlob(clean_text[0])
        pol = text.sentiment.polarity
        if pol > 0:
            positive["data"].append(clean_text)
            positive["polarity"].append(pol)
        elif pol == 0:
            neutral["data"].append(clean_text)
            neutral["polarity"].append(pol)
        elif pol < 0:
            negative["data"].append(clean_text)
            negative["polarity"].append(pol)
    polarity = {"positive": positive,
                "negative": negative, "neutral": neutral}
    return polarity


def main():
    path = r"C:\Users\Kazuki\thesis\data"
    os.chdir(path)
    random.seed(726)

    r = RandomWords()
    randomcount = [random.randint(3, 150) for i in range(10000)]
    docs = []
    for k in randomcount:
        word = r.get_random_words(limit=k)
        randomtext = [" ".join(word)]
        docs.append(randomtext)

    length = len(docs)
    sentiment = get_sentiment(docs)
    positive_per = len(sentiment["positive"]["data"]) / length * 100
    negative_per = len(sentiment["negative"]["data"]) / length * 100
    neutral_per = len(sentiment["neutral"]["data"]) / length * 100
    positive_pol = sum(sentiment["positive"]["polarity"]) / length
    negative_pol = sum(sentiment["negative"]["polarity"]) / length

    print("感情分析完了しました")
    print("Positive comments percentage: {} %".format(positive_per))
    print("Negative comments percentage: {} %".format(negative_per))
    print("Neutral comments percentage: {} %".format(neutral_per))
    print("Positive polaritys : {} ".format(positive_pol))
    print("Negative polaritys : {} ".format(negative_pol))


if __name__ == "__main__":
    main()
