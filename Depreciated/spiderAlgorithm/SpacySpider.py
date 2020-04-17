import pandas as pd
import json
import operator

import spacy
nlp = spacy.load('en')


# reading the categories into "categories"
def getCategories():
    #'/home/jscharf/Research/polisciresearch/misc/categories.txt'
    with open ('/home/jscharf/Research/polisciresearch/vars/categories.txt', 'r') as f:
        categories = f.read().split()
    return categories


# read categories synonyms into "categoriesRaw"
# @param String category - the category to search the json file for
def getCategorySynonyms(category):
    #'/home/jscharf/Research/polisciresearch/misc/categories.json'
    with open('/home/jscharf/Research/polisciresearch/vars/categories.json', 'r') as f:
        categoriesRaw = json.load(f)
    # assuring that the passed category is an actual category
    try:
        return categoriesRaw[category]
    except:
        return [category]



#Build categories into 2D list
def assembleCategories():
    cats = getCategories()

    result = []

    for c in cats:
        result.append(getCategorySynonyms(c))

    return result

assembleCategories()

class SpacySpider:

    def __init__(self):
        #Load up external categories
        self.categoryNames = getCategories()
        self.categories = assembleCategories()
    
    #loop through all special terms of category list
    #return score
    def __compareCategory(self, text, category):
        phrase = nlp(text.lower())
        
        scores = []
        
        #Compare with seed words
        for seedWord in category:
            scores.append(nlp(seedWord).similarity(phrase))
        
        return self.__score(scores)
    
    #I am not sure how we should pick the score, so rn it's just an average basically
    def __score(self, scores):
        total = 0
        for x in scores:
            total = total + x

        return total/len(scores)
    
    #Generate a score in every category
    def __scoreAll(self, text):
        scores = []
        for catSeeds in self.categories:
            scores.append(self.__compareCategory(text, catSeeds))

        return scores

    #generate scores and pick the max category
    def categorizePhrase(self, phrase):
        scores = self.__scoreAll(phrase)
        return self.categoryNames[scores.index(max(scores))]


