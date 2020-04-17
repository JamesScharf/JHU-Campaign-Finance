# Spider Algorithm
# Updated Feb 3, 2019
# Recent changes:
# Complete class encapsulation, python optimization, cleaning
import pandas as pd
from datamuse import datamuse
import json
import operator
import pickle
import copy

api = datamuse.Datamuse()

class FiveByFiveAlgorithm:
    '''
    self.categories : contains the name of each category (String[])
    self.seedWords : contains the seed words of each category (String[][])
    self.expanded : contains all datamuse-expanded words of each seedWord (String[][])
    '''
    '''
    self.TIMES_TO_EXPAND : # the number of times to iteratively expand each word
    self.GET_TOP_N :  the top number to get, is five because five_by_five
    '''


    def __init__(self, useSaved = False):
        # Global Static variables
        self.TIMES_TO_EXPAND = 1
        self.GET_TOP_N = 2

        # create new algorithm using categories.json
        if (useSaved): # when wanting to use pre-generated database information
            data = pickle.load(open("fivebyfive.p", "rb"))
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
        pickle.dump([self.categories, self.seedWords, self.expanded], open("fivebyfive.p", "wb"))


    def __datamuseSeedExpansion(self): 
        # specific application of __datamuseExpansion for seed words
        self.expanded = []
        listToExtend = 0
        for categorySynonymList in self.seedWords:
            print("Expanding List " + str(listToExtend) + " of " + str(len(self.seedWords)))
            self.expanded.append([])
            self.expanded[listToExtend].extend(self.__datamuseExpansion(categorySynonymList))
            listToExtend += 1

    def __datamuseExpansion(self, wordArr): 
        # expands an String[] into similar words
        expandedWords = []
        for i in range (self.TIMES_TO_EXPAND): # repeat this process n times
            if (i != 0): # on later iterations, replace the wordArr (which is being expanded) ...
                # ... with previous expansion results
                wordArr = copy.deepcopy(expandedWords)
            for word in wordArr:
                expandedWords.append(word) # in the expanded words add the word itself
                newWords = self.__getWordsFromAPI(word)
                expandedWords.extend(newWords) # add also the expansion of each word
            expandedWords = list(set(expandedWords)) # remove duplicate words
        return expandedWords
    
    def __getWordsFromAPI(self, word):
        # wrapper function for API call
        wordDict = api.words(ml = word)
        APIWords = []
        count = 0
        for dictEntry in wordDict:
            if (count >= self.GET_TOP_N): # only get the top n results
                break
            APIWords.append(dictEntry["word"])
            count += 1
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

    def categorizePhrase(self, phrase, expandPhrase = False):
        # public function to categorize a word
        phrase = self.__cleanPharse(phrase)
        phrase = phrase.split(" ") # splitting by word
        phraseExpanded = [] # expand phrase into datamuse

        # if the user wants to also expand the phrase
        if (expandPhrase): 
            for word in phrase:
                phraseExpanded.extend(self.__datamuseExpansion([word]))

        # if the user wants to not expand the phrase        
        else: 
            phraseExpanded = phrase
        

        scores = [] # scores in each category
        for categoryExpandedArr in self.expanded:
            # score similarity between categoryExpanedd and the purposeCode
            scores.append(self.__scoreSimilarity(phraseExpanded, categoryExpandedArr))
        # get name of category with max score
        return (self.categories[scores.index(max(scores))])

