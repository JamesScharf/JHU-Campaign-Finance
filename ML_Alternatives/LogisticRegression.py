'''
Note to programmer: If you want to test, remove "##" from lines and comment
out the following line.
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
import numpy as np
from nltk import word_tokenize
import datamuse
import json

vectorizer = TfidfVectorizer(use_idf = False, max_features=4250)

stop_words = set(stopwords.words('english')) 


def preprocess(phrase):
    phrase = phrase.lower()
    phrase = word_tokenize(phrase)
    punct = "`~!@#$%^&*()_+=|\}]{[':;?/>.<,"
    nums = "1234567890"
    phrase = [w.upper() for w in phrase if not w in stop_words]
    phrase = [w.upper() for w in phrase if not w in punct]
    phrase = [w.upper() for w in phrase if not w in nums]
    #phrase = [stemmer.stem(w) for w in phrase]
    #tb = TextBlob(" ".join(phrase))
    #phrase = tb.correct().split()

    tempPhrase = phrase.copy()
    for word in tempPhrase:
        if word in catWords:
            phrase.extend(word*15)
        if word in synonyms:
            phrase.extend(word*5)

    return " ".join(phrase)

def categorySet():
    '''
    Load up all of the categories into one big array
    '''
    categories = json.load(open("../vars/categories.json"))

    #loop through and make flat list of all
    catSet = set()
    for cat in categories:
        toAdd = []
        for term in categories[cat]:
            temp = term.upper()
            toAdd.extend(temp.split())
        catSet.update(toAdd)
    
    return catSet

api = datamuse.Datamuse()

def getWord(word):
    '''
    Datamuse
    '''
    wordList = api.words(ml = word)
    
    results = set()
    for wordDict in wordList:
        results.add(wordDict["word"].upper())
    return results

def synList(catSet):
    '''
    Generate a list of synonyms for every word in the catSet
    '''
    
    synSet = set()
    for word in catSet:
        synSet.update(set(getWord(word)))
    return synSet

catWords = categorySet()
synonyms = synList(catWords)

def processData():
    
    unprocessed = pd.read_csv("comboTraining-train.tsv")
    #unprocessed = pd.read_csv("comboTraining.csv")
    corpus = unprocessed["Purpose Code"].astype(str).values

    #process
    corpus = [preprocess(phrase) for phrase in corpus]
    X = vectorizer.fit_transform(corpus).toarray()
    global scaler
    scaler = MaxAbsScaler().fit(X)
    X = scaler.transform(X)

    tempy = unprocessed["Categorization"]
    valsToNums = unprocessed["Categorization"].unique().tolist()
    for index, vals in enumerate(valsToNums):
        tempy = tempy.replace(vals, index)

    y = tempy.values
    return X, y, valsToNums

def makePrediction(clf, vectorized, valsToNums):
    vectorized = scaler.transform(vectorized)
    prediction = clf.predict(vectorized)
    return valsToNums[prediction[0]]


def test(clf, valsToNums):
    #The testing portion
    test = pd.read_csv("comboTraining-dev.tsv")
    #process the testing data
    for index, vals in enumerate(valsToNums):
        test = test.replace(vals, index)
    correct = 0.0

    pCodes = test["Purpose Code"].astype(str).values
    correctCats = test["Categorization"].astype(str).values

    for testX, testY in zip(pCodes, correctCats):
        tempX = vectorizer.transform([testX])
        predicted = makePrediction(clf, tempX, valsToNums)
        
        if predicted == valsToNums[int(testY)]:
            correct += 1

    print(correct / len(test))

def categorizePhrase(phrase):
    phrase = preprocess(phrase)
    tempX = vectorizer.transform([phrase])
    predicted = makePrediction(clf, tempX, valToNums)
    return predicted

X, y, valToNums = processData()
global clf
clf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial')
clf.fit(X, y) 

test(clf, valToNums)