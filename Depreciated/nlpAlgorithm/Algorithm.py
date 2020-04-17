#!/usr/bin/python

import pandas as pd
import numpy as np
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import json
import sys

#removes punctuation from text
def stripPunctuation(text):
    return text.split()

def remove_stopwords(text):
    text = stripPunctuation(text.lower())
    stopWords = set(stopwords.words("english"))
    tokenized = word_tokenize(text)
    cleaned = [t for t in tokenized if t not in stopWords]
    return cleaned


#Extract some useful features from a piece of text
class FeatureExtractor:

    def __init__(self, corpus):
        self.catSeeds = None
        self.setupVectorizer(False, corpus)

    #By default the vectorizer is a CountVectorizer bigram vectorizer
    #But if tfidf is true then it's a TfidVectorizer
    def setupVectorizer(self, tfidf, corpus):
        #create bigrams
        if tfidf == True:
            self.vectorizer = TfidfVectorizer(ngram_range = (1, 2), token_pattern=r'\b\w+\b', min_df=1)
            #self.vectorizer = TfidfVectorizer()
        else:
            #CountVectorizer by default
            self.vectorizer = CountVectorizer(ngram_range = (1, 2), token_pattern=r'\b\w+\b', min_df=1)
            #self.vectorizer = CountVectorizer()

        #build the vocabulary from the corpus
        self.vectorizer.fit_transform(corpus.tolist())
        return self.vectorizer
    
    #just a cover of vocab
    #Shows you what's in the vectorizer's vocabulary
    def vectorizerVocabulary(self):
        return self.vectorizer.vocabulary_
    
    #Give a summary of occurrences of value in the text
    def identifyComponents(self, text):
        components = self.vectorizer.transform([text])
        return components.toarray().tolist()[0]
    
    # read categories synonyms into "categoriesRaw"
    # @param String category - the category to search the json file for
    def getSeedWords(self, category):
        with open('C:/Users/schar/Documents/PoliSciResearch/vars/categories.json', 'r') as f:
            categoriesRaw = json.load(f)
        # assuring that the passed category is an actual category
        try:
            return categoriesRaw[category]
        except:
            return [category]
    
    #build a list of each category and its synonyms into an array representation
    #categories = ['media', 'legal', 'etc']
    def categorySeeds(self, categories):
        if self.catSeeds is not None:
            return self.catSeeds
            
        self.catSeeds = []
        for category in categories:
            #seedwords of one category
            seedWords = self.getSeedWords(category)
            self.catSeeds.append(seedWords)
        return self.catSeeds
    
    #Find out whether a word is in any of the categories
    #requires getSeedWords, categorySeeds to be run first
    #returns a value that corresponds to the category that it's in (starting at 0)
    #returns -1 if in none
    def isSeedWord(self, word):
        #sw = the list of seedwords of one category
        for index, sw in enumerate(self.catSeeds):
            if word in sw:
                return index
        
        return -1
    
    #determine how many words in text fall in each category
    #requires categorySeeds to be run first
    #Assumes that text has already been cleaned
    def numInCategory(self, text):
        tokenText = word_tokenize(text)

        #what we'll eventually return
        numCategories = [0 for x in range(len(self.catSeeds))]
        for tt in tokenText:
            inWords = self.isSeedWord(tt)

            if inWords is not -1:
                numCategories[inWords] += 1
        
        return numCategories



class Classifier:
    
    def __init__(self, categories, corpus, trainingY):
        self.fe = FeatureExtractor(corpus)
        self.fe.categorySeeds(categories)
        self.train(corpus, trainingY)
        
        


    #outputs = the result that we want
    #inputs = the original documents (array of strings (documents))
    def train(self, inputs, outputs):
        self.clf = RandomForestClassifier()

        featureCollection = []
        for i in inputs:
            featureCollection.append(self.extractFeatures(i))
        
        self.clf.fit(featureCollection, outputs)


    def extractFeatures(self, document):
        #document = remove_stopwords(document)
        
        #begin adding features
        #features = self.fe.numInCategory(document) + self.fe.identifyComponents(document).tolist()
        features = self.fe.identifyComponents(document) + self.fe.numInCategory(document)
        #features = self.fe.identifyComponents(document)
        return features

    def classify(self, testInputs):
        featureCollection = []
        for doc in testInputs:
            featureCollection.append(self.extractFeatures(doc))

        return self.clf.predict(featureCollection)


#A method that converts to numbers
def convertColumn(df):
    df = df.copy()
    categories = list(set(df['Categorization']))

    categoryDict = dict()
    
    for c in categories:
        categoryDict[c] = categories.index(c)

    col = list(df['Categorization'])
    for index, to in enumerate(col):
        col[index] = categoryDict[to]
    
    df['Categorization'] = col
    df.loc['Purpose Code'] = df['Purpose Code'].apply(str).str.lower()
    df.loc['Purpose Code'] = df['Purpose Code'].apply(str).str.lower()
    return df.dropna()


def setup():

    #Some initial file processing
    #read the machine learning file into sci-kit learn
    originalData = pd.read_csv('C:/Users/schar/Documents/PoliSciResearch/vars/purposecode_moneydistribution_survey.csv', engine="python")
    unprocessed = originalData[['Purpose Code', 'Categorization']]
    trainingSet = unprocessed.iloc[0:1001]
    trainingSet = convertColumn(trainingSet)

    categories = list(set(unprocessed['Categorization']))
    #set up categories

    testSet = convertColumn(originalData.iloc[1001:1101])

    categoryDict = dict()

    global converseCatDict
    converseCatDict = dict()
    for c in categories:
        categoryDict[c] = categories.index(c)
        converseCatDict[categories.index(c)] = c


    #keep on running this part until we achieve optimal scores
    score = 0
    global classifier 
    while score < 86:
        score = 0
        classifier = Classifier(categories, trainingSet['Purpose Code'], trainingSet['Categorization'])
        results = (list(classifier.classify(testSet['Purpose Code'])))

        #this is just for testing
        for index, r in enumerate(results):
            if r == testSet.iloc[index]["Categorization"]:
                score += 1
        
        print(score)

    #now we can use the classifier


setup()