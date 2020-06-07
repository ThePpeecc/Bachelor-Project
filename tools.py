""" 
    This document contains lots of small methods and functions which jobs is mostly to 
    simplify and standardize the data processing.
"""

import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from loader import LoadStopWords
from sklearn.feature_extraction import text
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
import gensim.corpora as corpora
from gensim.models import CoherenceModel

def sentenceSplitter(sent):
    # Splits a sentence into individual tokens
    tokenizer = RegexpTokenizer(r'\w+')
    splitted = tokenizer.tokenize(sent.lower())

    return np.asarray(list(filter(lambda x: x != '', splitted)))

def stopWordRemover(sent, useNLTK = False):
    # Removes stopwords for bow sentence
    sWords = LoadStopWords()
    if useNLTK:
        sWords = stopwords.words('danish')
    return np.asarray(list(filter(lambda x: x not in sWords, sent)))

def tokenize(data, useNLTK = False):
    # Tokenizes a list of sentences into a list of bow sentences, but also removes stopwords in the process
    tokens = []
    for sent in data:
        tokens.append(stopWordRemover(sentenceSplitter(sent), useNLTK))
    return np.array(tokens)

def stemDocument(data):
    # Tales a list of bow sentence and stems all tokens
    stem = DanishStemmer().stem
    stemmed = []
    for sent in data:
        newSent = []
        for word in sent:
            newSent.append(stem(word))
        stemmed.append(newSent)
    return stemmed

def SetupDocument(data):
    return stemDocument(tokenize(data))

def flat(l):
    # Flattens a multi dim array by one dim, ie 2d array -> 1d array
    return [item for sublist in l for item in sublist]

def remerge(x):
    # Merges a list bow back into a sentences
    return [" ".join(s) for s in x]

def CreateVocab(document):
    # Document is list of bow sentences
    return {word: i for i, word in enumerate(np.unique(flat(document)))}


def EncodeSent(sent, vocab):
    # Word -> hot encoded vector
    # Vocab should be word -> id
    base = np.zeros(len(vocab.keys()))

    for word in sent:
        base[vocab[word]] = 1
    return base

def EncodeDocument(document):
    # one hot encodes a list of bow sentences
    vocab = CreateVocab(document)
    encoded = []
    for sent in document:
        encoded.append(EncodeSent(sent, vocab))
    return np.array(encoded)


def EncodeFT(ft, data):
    # Encode a list of bow sentence into fasttext sentence embeddings
    allSentenceVetors = []
    for sent in data:
        if len(sent) == 0: continue # Incase of empy strings
        sentenceVec = ft.get_sentence_vector(" ".join(sent)) 
        allSentenceVetors.append(sentenceVec)
    return np.array(allSentenceVetors)

def vocabCreater(tokens):
    # Take a list of bow sentences and create a vocabulary of the type token -> count
    vocab = {}
    for sent in tokens:
        for word in sent:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab

def getMoreSent(dataRaw):
    # Processes data and removes chars that can work as sentence splitters. 
    # This usally results in more sentences than without this processing set.

    # Remove digits
    dataRaw = ''.join(filter(lambda c: not c.isdigit(), dataRaw))
    # List of chars we want to manipulate or remove
    replace = [(' - ', ' '), ('-', '.'), ('?', '? '), ('!', '! '), (':', '.'), (';', '.'), ('...', '.'), ('..', '.'), ('.', '. ')]
    for change, to in replace:
        dataRaw = dataRaw.replace(change, to)

    batch = sent_tokenize(dataRaw)

    return np.unique(batch)

def getNTokens(data, grams = 2, joiner = '_'):
    # This method creates ngrams for a list of bow sentences
    t = [list(ngrams(sent, grams)) for sent in data]
    t = [[joiner.join(list(v)) for v in l] for l in t]
    return t

def getMoreTokens(data, grams = 2, joiner = '_'):
    # Creates extra tokens for a list of sentences by creating bi-grams
    t = getNTokens(data, grams, joiner)
    t = [list(l) + list(r) for l, r in list(zip(t, data))]
    return t


def rawTokenize(data):
    # Tokenizes but does not remove any information in the process ie stopwords
    tokens = []
    for sent in data:
        tokens.append(sentenceSplitter(sent))
    return np.array(tokens)

def cleanSent(sent, vocab_count):
    # Cleans a sentence for words that appear too much or too little
    nSent = []
    for word in sent:
        if vocab_count[word] > 2 and vocab_count[word] < 1000 and len(word) > 1:
            nSent.append(word)

    return nSent

def stemmedReverse(data):
    # Creates a reverse lookup table to lookup the origins of stemmed words ie stemmed -> words
    stemF = DanishStemmer().stem
    words_raw = np.unique(flat(data))
    stemmed = {}
    
    for word in words_raw:
        stem = stemF(word)
        if stem not in stemmed:
            stemmed[stem] = [word]
        else:
            stemmed[stem].append(word)
    
    return stemmed


def cleanDoc(data, minN = 0, maxN = 1000):
    # Cleans a document of sentences, and removes any sentence that are too long or too short
    newAr = []

    vocab = vocabCreater(data)

    vocab_count = {l: v for l, v in vocab.items()}
    for sent in data:
        nSent = cleanSent(sent, vocab_count)
        if len(nSent) >= minN and len(nSent) <= maxN:
            newAr.append(nSent)

    return newAr

def computeScore(topics, texts, id2word):
    # Computes the coherence score 
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=id2word, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence

def Id2Word(data):
    # Create Dictionary
    id2word = corpora.Dictionary(data)

    return id2word
