import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import MeCab
import nltk
import chardet
import codecs
import re
import os
from tqdm import tqdm
from glob import glob
from textblob import TextBlob


def preprocessing(text):
    no_space = re.compile(r"[.;:!\'?,\"()\[\]]")
    #with_space = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")
    number = re.compile(r"[0-9]+")
    text = [no_space.sub("", line.lower()) for line in text]
    text = [number.sub("0", line) for line in text]
    #text = [with_space(" ", line) for line in text]
    return text


def get_words(text, conditions=["FW", "JJ", "JJR", "JJS", "LS", "NN", "NNP", "RB", "RBR",
                                "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"], language="English"):
    """
    :param text: text data
    :param language: you can select language in [English,Japanese]
    :param conditions: you can select word class in Japanese or English
    :param wordcloud: you choice to print wordcloud
    $: dollar
        $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
    '': closing quotation mark
        ' ''
    (: opening parenthesis
        ( [ {
    ): closing parenthesis
        ) ] }
    ,: comma
        ,
    --: dash
        --
    .: sentence terminator
        . ! ?
    :: colon or ellipsis
        : ; ...
    CC: conjunction, coordinating
        & 'n and both but either et for less minus neither nor or plus so
        therefore times v. versus vs. whether yet
    CD: numeral, cardinal
        mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
        seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
        fifteen 271,124 dozen quintillion DM2,000 ...
    DT: determiner
        all an another any both del each either every half la many much nary
        neither no some such that the them these this those
    EX: existential there
        there
    FW: foreign word
        gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
        lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
        terram fiche oui corporis ...
    IN: preposition or conjunction, subordinating
        astride among uppon whether out inside pro despite on by throughout
        below within for towards near behind atop around if like until below
        next into if beside ...
    JJ: adjective or numeral, ordinal
        third ill-mannered pre-war regrettable oiled calamitous first separable
        ectoplasmic battery-powered participatory fourth still-to-be-named
        multilingual multi-disciplinary ...
    JJR: adjective, comparative
        bleaker braver breezier briefer brighter brisker broader bumper busier
        calmer cheaper choosier cleaner clearer closer colder commoner costlier
        cozier creamier crunchier cuter ...
    JJS: adjective, superlative
        calmest cheapest choicest classiest cleanest clearest closest commonest
        corniest costliest crassest creepiest crudest cutest darkest deadliest
        dearest deepest densest dinkiest ...
    LS: list item marker
        A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
        SP-44007 Second Third Three Two * a b c d first five four one six three
        two
    MD: modal auxiliary
        can cannot could couldn't dare may might must need ought shall should
    NN: noun, common, singular or mass
    common-carrier cabbage knuckle-duster Casino afghan shed thermostat
    investment slide humour falloff slick wind hyena override subhumanity
    machinist ...
    NNP: noun, proper, singular
        Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
        Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
        Shannon A.K.C. Meltex Liverpool ...
    NNPS: noun, proper, plural
        Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
        Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
        Apache Apaches Apocrypha ...
    NNS: noun, common, plural
        undergraduates scotches bric-a-brac products bodyguards facets coasts
        divestitures storehouses designs clubs fragrances averages
        subjectivists apprehensions muses factory-jobs ...
    PDT: pre-determiner
        all both half many quite such sure this
    POS: genitive marker
        ' 's
    PRP: pronoun, personal
        hers herself him himself hisself it itself me myself one oneself ours
        ourselves ownself self she thee theirs them themselves they thou thy us
    PRP$: pronoun, possessive
        her his mine my our ours their thy your
    RB: adverb
        occasionally unabatingly maddeningly adventurously professedly
        stirringly prominently technologically magisterially predominately
        swiftly fiscally pitilessly ...
    RBR: adverb, comparative
        further gloomier grander graver greater grimmer harder harsher
        healthier heavier higher however larger later leaner lengthier less-
        perfectly lesser lonelier longer louder lower more ...
    RBS: adverb, superlative
        best biggest bluntest earliest farthest first furthest hardest
        heartiest highest largest least less most nearest second tightest worst
    RP: particle
        aboard about across along apart around aside at away back before behind
        by crop down ever fast for forth from go high i.e. in into just later
        low more off on open out over per pie raising start teeth that through
        under unto up up-pp upon whole with you
    SYM: symbol
        % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
    TO: "to" as preposition or infinitive marker
        to
    UH: interjection
        Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
        huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
        man baby diddle hush sonuvabitch ...
    VB: verb, base form
        ask assemble assess assign assume atone attention avoid bake balkanize
        bank begin behold believe bend benefit bevel beware bless boil bomb
        boost brace break bring broil brush build ...
    VBD: verb, past tense
        dipped pleaded swiped regummed soaked tidied convened halted registered
        cushioned exacted snubbed strode aimed adopted belied figgered
        speculated wore appreciated contemplated ...
    VBG: verb, present participle or gerund
        telegraphing stirring focusing angering judging stalling lactating
        hankerin' alleging veering capping approaching traveling besieging
        encrypting interrupting erasing wincing ...
    VBN: verb, past participle
        multihulled dilapidated aerosolized chaired languished panelized used
        experimented flourished imitated reunifed factored condensed sheared
        unsettled primed dubbed desired ...
    VBP: verb, present tense, not 3rd person singular
        predominate wrap resort sue twist spill cure lengthen brush terminate
        appear tend stray glisten obtain comprise detest tease attract
        emphasize mold postpone sever return wag ...
    VBZ: verb, present tense, 3rd person singular
        bases reconstructs marks mixes displeases seals carps weaves snatches
        slumps stretches authorizes smolders pictures emerges stockpiles
        seduces fizzes uses bolsters slaps speaks pleads ...
    WDT: WH-determiner
        that what whatever which whichever
    WP: WH-pronoun
        that what whatever whatsoever which who whom whosoever
    WP$: WH-pronoun, possessive
        whose
    WRB: Wh-adverb
        how however whence whenever where whereby whereever wherein whereof why
    ``: opening quotation mark
        ` ``
    """

    words = []
    if language == "English":
        text = re.subn(r"\W", " ", text)[0]
        tokens = nltk.word_tokenize(text)
        stop_words = nltk.corpus.stopwords.words("english")
        change_tokens = [word for word in tokens if word not in stop_words]
        tagged = nltk.pos_tag(change_tokens)

        for tag in tagged:
            if tag[1].split(",")[0] in conditions:
                words.append(tag[0])

    elif language == "Japanese":
        t = MeCab.Tagger()
        text = re.subn(r"\W", "", text)[0]
        tokens = t.parse(text)
        tokens = [line.split("\t") for line in tokens.split("\n")][:-2]

        for i in tokens:
            if i[1].split(",")[0] in conditions:
                words.append(i[0])
    return words


def get_comments(file):
    with open(file, "rb") as f:
        file_code = chardet.detect(f.read())["encoding"]
    with codecs.open(file, 'r', file_code, 'ignore') as f:
        comment = pd.read_csv(f)
    comment = comment.dropna(how="any")
    return comment


def make_dictionary(docs):
    word2num = dict()
    num2word = dict()
    count = 0

    for d in docs:
        for w in d:
            if w not in word2num.keys():
                word2num[w] = count
                num2word[count] = w
                count += 1
    return word2num, num2word


def my_LDA(docs, word2num, K=5, Iter=1000, alpha=0.1, beta=0.1, trace=False, inter=10):
    def sampling(ndk, nkv, nd, nk, i, v):
        probs = np.zeros(K)
        for k in range(K):
            theta = (ndk[i, k] + alpha) / (nd[i] + alpha * M)
            phi = (nkv[k, v] + beta) / (nk[k] + beta * K)
            prob = theta * phi
            probs[k] = prob
        probs /= probs.sum()
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    np.random.seed(1000)

    V = len(word2num)  # <- number of words
    M = len(docs)      # <- number of documents
    ndk = np.zeros((M, K))  # topic distribution of sentences
    nkv = np.zeros((K, V))  # word disrtibution of each topics

    topics = [[np.random.randint(K) for w in d] for d in docs]
    for i, d in enumerate(topics):
        for j, z in enumerate(d):
            ndk[i, z] += 1
            nkv[z, docs[i][j]] += 1
    nd = ndk.sum(axis=1)
    nk = nkv.sum(axis=1)

    if trace:
        chain = []
    for ite in tqdm(range(Iter)):
        move = 0
        for i, d in enumerate(topics):  # Every Documents
            for j, k in enumerate(d):  # Every word and topics
                v = docs[i][j]
                ndk[i, k] -= 1
                nkv[k, v] -= 1
                nk[k] -= 1
                new_z = sampling(ndk, nkv, nd, nk, i, v)
                if trace and ite % inter == 0:
                    if new_z != k:
                        move += 1
                topics[i][j] = new_z
                ndk[i, new_z] += 1
                nkv[new_z, v] += 1
                nk[new_z] += 1
        if ite % inter == 0:
            chain.append(move)
    save = {"topics": topics, "nkv": nkv, "ndk": ndk, "nk": nk, "nd": nd}
    if trace:
        save["trace"] = chain
        print("word move:", chain)
        plt.xlabel("Iteration")
        plt.ylabel("Number of renewal topics")
        plt.plot(chain)
        plt.show()
    return save


def topwords(nkv, num2word, t=10):
    sphi = np.argsort(nkv, axis=1).T[::-1].tolist()
    topwords = [[num2word[i] for i in w] for w in sphi]
    return pd.DataFrame(topwords).iloc[:10, :]


def delete(path):
    files = glob("*.pkl")
    print("フォルダ内の.pklファイルをすべて削除しますか？(y/n)")
    confirm = input()
    while (confirm != "y") and (confirm != "n"):
        print("無効な値が入力されました。")
        print("もう一度入力してください。(y/n)")
        confirm = input()

    if confirm == "y":
        for file_name in files:
            os.remove(path + "\\" + file_name)


def confirmation():
    files = glob("*")
    print("ファルダ内のファイルをすべて確認しますか？(y/n)")
    confirm = input()
    while (confirm != "y") and (confirm != "n"):
        print("無効な値が入力されました。")
        print("もう一度入力してください。(y/n)")
        confirm = input()

    if confirm == "y":
        for file_name in files:
            print(file_name)


def read_pkl(file, label=None, value=None):
    file_root = os.path.splitext(file)[0]
    if (label != None) & (value != None):
        sentiment = True
    else:
        sentiment = False
    if sentiment:
        ndocs_file = file_root + "_" + label + "_ndocs.pkl"
        w2n_file = file_root + "_" + label + "_w2n.pkl"
        n2w_file = file_root + "_" + label + "_n2w.pkl"
    else:
        ndocs_file = file_root + "_ndocs.pkl"
        w2n_file = file_root + "_w2n.pkl"
        n2w_file = file_root + "_n2w.pkl"
    print("%s というファイルがあるか確認します" % ndocs_file)

    if os.path.isfile(ndocs_file):
        print("%s というファイルはありました" % ndocs_file)
        ndocs = pkl.load(open(ndocs_file, "rb"))
        word2num = pkl.load(open(w2n_file, "rb"))
        num2word = pkl.load(open(n2w_file, "rb"))
    else:
        print("%s というファイルはないので、作成します。" % ndocs_file)
        if sentiment:
            text = value
        else:
            df = get_comments(file)
            df = df.sort_index(ascending=False)
            text = preprocessing(df["comment"])
        docs = [get_words(comment) for comment in text]
        docs = list(filter(lambda x: x != [], docs))
        word2num, num2word = make_dictionary(docs)
        ndocs = [[word2num[w] for w in d] for d in docs]
        pkl.dump(ndocs, open(ndocs_file, "wb"))
        pkl.dump(word2num, open(w2n_file, "wb"))
        pkl.dump(num2word, open(n2w_file, "wb"))
        print("Complete!!")
    return ndocs, word2num, num2word


def get_sentiment(file):
    positive = []
    negative = []
    neutral = []
    df = get_comments(file)
    df = df.sort_index(ascending=False)
    clean_text = preprocessing(df["comment"])
    # 感情分類の手法はTextBlobを使っただけなので、変更可能
    # 感情値に関しては出していない
    for comment in clean_text:
        text = TextBlob(comment)
        if text.sentiment.polarity > 0:
            positive.append(comment)
        elif text.sentiment.polarity == 0:
            neutral.append(comment)
        elif text.sentiment.polarity < 0:
            negative.append(comment)

    positive_per = len(positive) / len(clean_text) * 100
    negative_per = len(negative) / len(clean_text) * 100
    neutral_per = len(neutral) / len(clean_text) * 100
    print("感情分析完了しました")
    print("Positive comments percentage: {} %".format(positive_per))
    print("Negative comments percentage: {} %".format(negative_per))
    print("Neutral comments percentage: {} %".format(neutral_per))

    porarity = {"positive": positive, "negative": negative, "neutral": neutral}
    return porarity


def result(ndocs, word2num, num2word):
    result = my_LDA(ndocs, word2num, K=5, trace=True)
    print("After sampling\n", result["topics"][:10])
    print("Top words\n", topwords(result["nkv"], num2word))


def main():
    path = r"C:\Users\Kazuki\Paper\GraduateThesis\data"
    os.chdir(path)
    confirmation()
    delete(path)
    files = glob("*.csv")

    sentiment = False

    for file in files:
        if sentiment:
            porarity = get_sentiment(file)
            for (label, data) in porarity.items():
                ndocs, word2num, num2word = read_pkl(file, label, data)
                result(ndocs, word2num, num2word)
        else:
            ndocs, word2num, num2word = read_pkl(file)
            result(ndocs, word2num, num2word)


if __name__ == "__main__":
    main()
