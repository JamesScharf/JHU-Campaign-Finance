# Spider Algorithm
# Updated Feb 3, 2019
# Recent changes:
# Complete class encapsulation, python optimization, cleaning
import pandas as pd
from datamuse import datamuse
import json
import operator
import pickle

api = datamuse.Datamuse()

class FireworkAlgorithm:
    '''
    self.categories : contains the name of each category (String[])
    self.seedWords : contains the seed words of each category (String[][])
    self.expanded : contains all datamuse-expanded words of each seedWord (String[][])
    '''

    def __init__(self, useSaved = False):
        # create new algorithm using categories.json
        if (useSaved): # when wanting to use pre-generated database information
            data = pickle.load(open("firework.p", "rb"))
            self.categories = data[0]
            self.seedWords = data[1]
            self.expanded = data[2]
            return
        with open('../vars/categories.json', 'r') as f:
            categoriesRaw = json.load(f)
            self.categories = []
            self.seedWords = []
            for category in categoriesRaw.keys():
                self.categories.append(category)
                self.seedWords.append(categoriesRaw[category])
        self.__datamuseSeedExpansion()
        pickle.dump([self.categories, self.seedWords, self.expanded], open("firework.p", "wb"))


    def __datamuseSeedExpansion(self): 
        # specific application of __datamuseExpansion for seed words
        self.expanded = []
        listToExtend = 0
        for categorySynonymList in self.seedWords:
            self.expanded.append([])
            self.expanded[listToExtend].extend(self.__datamuseExpansion(categorySynonymList))
            listToExtend += 1

    def __datamuseExpansion(self, wordArr): 
        # expands an String[] into similar words
        expandedWords = []
        for word in wordArr:
            expandedWords.append(word)
            expandedWords.extend(self.__getWordsFromAPI(word))
        expandedWords = list(set(expandedWords)) # remove duplicate words
        return expandedWords
    
    def __getWordsFromAPI(self, word):
        # wrapper function for API call
        wordDict = api.words(ml = word)
        APIWords = []
        for dictEntry in wordDict:
            APIWords.append(dictEntry["word"])
        return APIWords

    def __scoreSimilarity (self, wordArray1, wordArray2):
        # score similarity between two sets
        set1 = set(wordArray1) # casting to set
        set2 = set(wordArray2) # casting to set
        intersectionSize = len(set1 & set2)
        unionSize = len(set1 | set2)
        return (round(intersectionSize / unionSize, 3))

    def __cleanPharse(self, phrase):
        # cleaning / preprocessing of a phrase
        phrase = phrase.lower()
        return phrase

    def categorizePhrase(self, phrase):
        # public function to categorize a word
        phrase = self.__cleanPharse(phrase)
        phrase = phrase.split(" ") # splitting by word
        phraseExpanded = [] # expand phrase into datamuse
        for word in phrase:
            phraseExpanded.extend(self.__datamuseExpansion([word]))

        scores = [] # scores in each category
        for categoryExpandedArr in self.expanded:
            # score similarity between categoryExpanedd and the purposeCode
            scores.append(self.__scoreSimilarity(phraseExpanded, categoryExpandedArr))
        # get name of category with max score
        return (self.categories[scores.index(max(scores))])
