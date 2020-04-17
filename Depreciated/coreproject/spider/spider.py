# Spider Algorithm
# Updated Feb 3, 2019
# Recent changes:
# Complete class encapsulation, python optimization, cleaning
import pandas as pd
from datamuse import datamuse
import json
import operator

api = datamuse.Datamuse()

class SpiderAlgorithm:

    #self.categories = [] # contains the name of each category
    #self.seedWords = [][] # contains the seed words of each category
    # each first list is for a category
    #self.expandedWithScores = [][] # contains all expanded words after datamuse
    # each first list is of a category
    #self.expanded = [][] # contains all datamuse-expanded words without scores
    # each first list is of a category

    # using categories.json, reads in the list of the categories
    # and the list of the seed words
    def __init__(self, useSaved = False):
        if (useSaved): # when wanting to use pre-generated database information
            return
        with open('../vars/categories.json', 'r') as f:
            categoriesRaw = json.load(f)
            self.categories = []
            self.seedWords = []
            for category in categoriesRaw.keys():
                self.categories.append(category)
                self.seedWords.append(categoriesRaw[category])
        self.__datamuseExpansion() # generating datamuse words
        self.__disjoint() # separating categories

    # populates two instance variables (expandedWithScores and expanded) will all datamuse words
    # of the synonyms
    def __datamuseExpansion(self):
        self.expandedWithScores = [] # all synonyms with scores
        self.expanded = [] # all synonyms without scores
        listToExtend = 0 # tracking array
        for categorySynonymList in self.seedWords:
            count = 0 # ensuring only taking top 40 results
            self.expandedWithScores.append([])
            self.expanded.append([])
            for synonym in categorySynonymList: # we supply category Synonym List
                if (count > 40): 
                    break
                self.expandedWithScores[listToExtend].extend([{'word': synonym, "score": 9999999, 'tags':[]}])
                self.expandedWithScores[listToExtend].extend(self.__getWordAndScoreFromAPI(synonym))
                self.expanded[listToExtend].extend(self.__getWordsFromAPI(synonym))
                self.expanded[listToExtend].append(synonym)
                count += 1
            listToExtend += 1

    # takes a string word and returns a dictionary of similar words with score category
    def __getWordAndScoreFromAPI(self, word):
        wordDict = api.words(ml = word)
        return wordDict

    # takes a string word and returns array of similar words
    def __getWordsFromAPI(self, word):
        wordDict = api.words(ml = word)
        APIWords = []
        for dictEntry in wordDict:
            APIWords.append(dictEntry["word"])
        return APIWords

    # ensuring each category is disjoint
    def __disjoint(self):
        # implementation not yet completed
        # perhaps there is a union find approach
        pass

    # helper method to score similarity between two arrays
    def __scoreSimilarity (self, wordArray1, wordArray2):
        set1 = set(wordArray1) # casting to set
        set2 = set(wordArray2) # casting to set
        intersectionSize = len(set1 & set2)
        unionSize = len(set1 | set2)
        return (round(intersectionSize / unionSize, 3))

    # cleaning / preprocessing of a phrase
    def __cleanPharse(self, phrase):
        phrase = phrase.lower()
        # function not yet implemented
        return phrase

    # public method to see sub-calculations
    def diagnostic(self):
        '''
        for x in range (len(self.categories)):
            print("Category: " + self.categories[x])
            print(list(set(self.expanded[x])))
            print("Total Unique Words: " + str(len(set(self.expanded[x]))))
            print("Total Duplicates: " + str(len(self.expanded[x]) - len(set(self.expanded[x]))))
            print("Total Seeds: " + str(len(self.seedWords[x])))
            print("Total Unique Words: " + str(len(set(self.expanded[x]))))
            print()
            print()
        '''
        for x in range (len(self.categories)):
            for y in range (len(self.categories)):
                print("Category 1: " + self.categories[x])
                print("Category 2: " + self.categories[y])
                totalWords = list(set(self.expanded[x])) + list(set(self.expanded[y]))
                print("Total Size: " + str(len(totalWords)))
                duplicates = len(totalWords) - len(list(set(totalWords)))
                print("Duplicates: " + str(duplicates))
                print(round(duplicates / len(totalWords), 3))
                print()
                print()
        pass

    # over-encompasing function to categorize a word
    def categorizePhrase(self, phrase):
        phrase = self.__cleanPharse(phrase)
        phrase = phrase.split(" ") # splitting by word
        # by this point, the datamuse-similar words have been generated
        # thus we can use the score similarity to get the similarity to each set of words
        scores = [] # scores to each category
        for totalWordList in self.expanded:
            scores.append(self.__scoreSimilarity(phrase, totalWordList))
        return (self.categories[scores.index(max(scores))])


s = SpiderAlgorithm()
print(s.categorizePhrase("MEDIA BUY"))
s.diagnostic() 
