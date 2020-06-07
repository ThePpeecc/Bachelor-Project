"""
    This file contains a function that will load the main text file and split it by sentences, and then return that array
"""
from io import open
import os
from nltk.tokenize import sent_tokenize
import fasttext
import fasttext.util

def LoadData(split='.'):
    with open('raw.txt', 'r') as file:
        data = file.read().replace('\n', '')
        data = sent_tokenize(data)
    file.close()

    return data

def loadDataParticipants():
    participants = []
    res = os.listdir('participants')
    res.sort(key = lambda x: int(x.replace(".txt", "")))
    for filename in res:    
        with open(os.path.join('participants', filename), 'r') as file: # open in readonly mode
            data = file.read().replace('\n', ' ')
            data = sent_tokenize(data)
        participants.append(data)
        file.close() 
    return participants

def LoadRaw():
    with open('raw.txt', 'r') as file:
        data = file.read().replace('\n', '')
    file.close()

    return data

def LoadStopWords():
    with open('stopwords.txt', 'r') as file:
        data = file.read()
        data = data.split('\n')
    file.close()

    return {word: True for word in data}


def LoadStemmings():
    with open('stemmings.txt', 'r') as file:
        data = file.read()
        data = data.split('\n')
    file.close()
    return {word: True for word in data}

def LoadWordEmbeddings(dim = 300):
    ft = fasttext.load_model('cc.da.300.bin')
    print("----- DONE LOADING -----")
    if ft.get_dimension() != dim:
        print("REDUCING DIMENSIONS")
        fasttext.util.reduce_model(ft, dim)
        print("DONE REDUCING")
    return ft
