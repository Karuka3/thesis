from nltk.corpus import words
from nltk.corpus import wordnet
import random


class RandomWords(object):
    def __init__(self):
        self.wordslist = words.words()

    def get_random_word(self, includePartOfSpeech=None, excludePartOfSpeech=None, minLength=1, maxLength=10):
        """
        Returns a single random word
        Args:
            includePartOfSpeech, str: CSV part-of-speech values to include (optional)
            excludePartOfSpeech, str: CSV part-of-speech values to exclude (optional)
            minCorpusCount, int: Minimum corpus frequency for terms (optional)
            maxCorpusCount, int: Maximum corpus frequency for terms (optional)
            minLength, int: Minimum word length (optional)
            maxLength, int: Maximum word length (optional)
        Returns: String, Random words
        """
        random.shuffle(self.wordslist)
        wordlist = [w for w in self.wordslist if minLength <=
                    len(w) <= maxLength]
        wordlist = wordlist[:10]
        word = wordlist[random.randint(0, 10)]
        return word

    def get_random_words(self, includePartOfSpeech=None, excludePartOfSpeech=None, minLength=2, maxLength=20, limit=10):
        """
        Returns a single random word
        Args:
            includePartOfSpeech, str: CSV part-of-speech values to include (optional)
            excludePartOfSpeech, str: CSV part-of-speech values to exclude (optional)
            minCorpusCount, int: Minimum corpus frequency for terms (optional)
            maxCorpusCount, int: Maximum corpus frequency for terms (optional)
            minLength, int: Minimum word length (optional)
            maxLength, int: Maximum word length (optional)
        Returns: String, Random words
        """
        random.shuffle(self.wordslist)
        wordlist = [w for w in self.wordslist if minLength <=
                    len(w) <= maxLength]
        words = wordlist[:limit]
        return words
