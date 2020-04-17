import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

originalData = pd.read_csv('C:/Users/schar/Documents/PoliSciResearch/vars/purposecode_moneydistribution_survey.csv', engine="python")
df = originalData[['Purpose Code', 'Categorization']]
trainingDocs = df.iloc[0:1001]
trainingDocs = trainingDocs["Purpose Code"]


class FeatureExtractor:

    def __init__(self):
        self.categorySeeds(["media", "digital", "polling", "legal", "field", "consulting", "administrative", "fundraising"])



    def extract_features(self, doc):
        features = []
        #features.extend(self.percentInCategory(doc))
        features.extend(self.phraseSimAll(doc))
        return features

    def getSeedWords(self, category):
        '''
        read categories synonyms into "categoriesRaw"
        @param String category - the category to search the json file for
        '''

        with open('C:/Users/schar/Documents/PoliSciResearch/vars/categories.json', 'r') as f:
            categoriesRaw = json.load(f)
        # assuring that the passed category is an actual category
        try:
            return categoriesRaw[category]
        except:
            return [category]
    
    def categorySeeds(self, categories):
        '''
        #build a list of each category and its synonyms into an array representation
        #categories = ['media', 'legal', 'etc']
        '''
        self.categoryNames = categories
        self.catDict = dict()
        self.catSeeds = []
        for category in categories:
            #seedwords of one category
            seedWords = self.getSeedWords(category)
            self.catDict[category] = seedWords
            self.catSeeds.append(seedWords)
                    

        return self.catSeeds
    
    def isSeedWord(self, word):
        '''
        Find out whether a word is in any of the categories
        requires getSeedWords, categorySeeds to be run first
        returns a value that corresponds to the category that it's in (starting at 0)
        returns -1 if in none
        '''
        #sw = the list of seedwords of one category
        for index, sw in enumerate(self.catSeeds):
            if word in sw:
                return index
        
        return -1

    def percentInCategory(self, text):
        '''
        determine how many words in text fall in each category
        requires categorySeeds to be run first
        Assumes that text has already been cleaned
        '''
        tokenText = word_tokenize(text)

        #what we'll eventually return
        numCategories = [0.0 for x in range(len(self.catSeeds))]
        for tt in tokenText:
            inWords = self.isSeedWord(tt)

            if inWords is not -1:
                numCategories[inWords] += 1.0
        
        percents = []
        splitText = text.split()
        textLength = len(splitText)
        for n in numCategories:
            percents.append(n / textLength)
        
        return percents

    def word_similarity(self, word, category):
        '''
        calculate similarity of one word to each category
        returns as a 2d array
        '''

        similarities = []

        if len(wn.synsets(word)) > 0:
            wordSyn = wn.synsets(word)[0]
            catSims = []
            category = self.catDict[category]
            for seed in category:
                if len(wn.synsets(seed)) > 0:
                    seedSyn = wn.synsets(seed)[0]
                    if (wordSyn.path_similarity(seedSyn)) is None:
                        catSims.append(0)
                    else:
                        catSims.append(wordSyn.path_similarity(seedSyn))
            
            similarities.append(catSims)
        
        #fix the small arrays
        temp = similarities.copy()
        for s in temp:
            if len(s) == 0:
                similarities.remove(s)
        return similarities


    def centroidCategory(self, similarities):
        '''
        Calculate the centroid of the a category
        So average each category to create a profile
        '''
        col = 0
        if len(similarities) == 0:
            return []
        rowLen = len(similarities[0])
        profile = []

        while(col < rowLen):
            profile.append(self.averageColumn(similarities, col))
            col += 1
        return profile

    def averageColumn(self, matrix, colNum):

        total = 0.0
        for row in matrix:
            total += row[colNum]
        return total/len(matrix)

    def phrase_similarity(self, phrase, category):
        '''
        Calculate the centroids (average of each column) of a phrase to a category
        '''

        #make an array where each row is a word's similarity
        phrase = phrase.split()
        sims = []
        for word in phrase:
            wordSim = self.word_similarity(word, category)
            if len(wordSim) > 0:
                sims.append(wordSim[0])
        
        return sims

    def phraseSimAll(self, phrase):
        '''
        generate scores and centroids for all categories
        and combine into one flat array (that can be plotted
        if a PCA is run on it)

        Use it to compare to category
        '''

        sims = []
        for category in self.categoryNames:
            sims.extend(self.centroidCategory(self.phrase_similarity(phrase, category)))

        return sims