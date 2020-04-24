'''
Note to programmer: If you want to test, remove "##" from lines and comment
out the following line.
'''

import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
import numpy as np
from nltk import word_tokenize
import datamuse
import json
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer(use_idf = False, analyzer="word", binary=True, max_features=4000)

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
            phrase.extend(word*25)
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
    
    ##unprocessed = pd.read_csv("comboTraining-train.tsv")
    unprocessed = pd.read_csv("comboTraining.csv")
    corpus = unprocessed["Purpose Code"].astype(str).values

    #process
    corpus = [preprocess(phrase) for phrase in corpus]
    X = vectorizer.fit_transform(corpus).toarray()

    tempy = unprocessed["Categorization"]
    valsToNums = unprocessed["Categorization"].unique().tolist()
    for index, vals in enumerate(valsToNums):
        tempy = tempy.replace(vals, index)

    y = tempy.values
    return X, y, valsToNums

def makePrediction(clf, vectorized, valsToNums):
    prediction = clf.predict(vectorized)
    return valsToNums[prediction[0]]


def test(clf, valsToNums):
    #The testing portion
    test = pd.read_csv("comboTraining-test.tsv")
    #process the testing data
    for index, vals in enumerate(valsToNums):
        test = test.replace(vals, index)
    correct = 0.0

    pCodes = test["Purpose Code"].astype(str).values
    correctCats = test["Categorization"].astype(str).values
    unProcPredictedCats = []
    for testX, testY in zip(pCodes, correctCats):
        tempX = vectorizer.transform([testX])
        predicted = makePrediction(clf, tempX, valsToNums)
        unProcPredictedCats.append(predicted)
        if predicted == valsToNums[int(testY)]:
            correct += 1

    #translate back to numbers
    predictedCats = []
    #build map of values
    mapCats = dict()
    for index, x in enumerate(valsToNums):
        mapCats[x] = index
    for x in unProcPredictedCats:
        predictedCats.append(mapCats[x])
    
    procCorrected = []
    for y in correctCats:
        procCorrected.append(int(y))
    print(classification_report(procCorrected, predictedCats, target_names=valsToNums))

def categorizePhrase(phrase):
    phrase = preprocess(phrase)
    tempX = vectorizer.transform([phrase])
    predicted = makePrediction(clf, tempX, valToNums)
    return predicted

X, y, valToNums = processData()
global clf
clf = linear_model.SGDClassifier(max_iter=2000, tol=1e-3, loss="hinge")
clf.fit(X, y) 

##test(clf, valToNums)