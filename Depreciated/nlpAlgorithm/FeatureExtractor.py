#Last edited 2/13/19 by James Scharf

#Goal of the document: Extract features
#                       Maybe do morpheme stuff and things with brown corpus

#Conventions: Each function should return some numerical value in a LIST
#               Using a list allows us to incorporate 

#Planned resources to use:
# polyglot (for morphemes)
# nltk (for bigrams, tokenization, misc random stuff)
# spacy (for lexemes) https://spacy.io/api/lexeme



import spider
from nltk.tokenize import word_tokenize
#Usage: "for word in word_tokenize(some_sentence)"


import pandas as pd
import json
import operator
import Similarity
import WordExpander
import statistics
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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



#useful: https://www.nltk.org/book/

class FeatureExtractor():
    
    def __init__(self):
        self.terms = assembleCategories   
        self.we = WordExpander.WordExpander()
        self.we.loadCorpus()
        self.Similarity = Similarity.Similarity()
        self.spider = spider.SpiderAlgorithm()

    def __tokenize(self, text):
        return word_tokenize(text)


        return results
    
    def extract(self, text):
        expandedText = " ".join(self.we.expand(text, True, True, True))
        

        features = [
            len(text),
            len(expandedText),
            1 if 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 in text else 0,
            hash(self.spider.categorizePhrase(text))
        ]

        features.extend(self.we.freqCorpus(expandedText))
        return features 
    
    #run extract on an array of strings
    def extractAll(self, textList):
        result = []
        for tl in textList:
            result.append(self.extract(tl))
            
        return result
    
    #standardize data
    #takes a list of strings as input
    def standardizeData(self, stringList):
        data = self.extractAll(stringList)
        nData = np.array(data)
        dfData = pd.DataFrame(nData)
        dfData = StandardScaler().fit_transform(dfData)
        return dfData
    
    #get PCA data
    def pca(self, stringList):
        dfData = self.standardizeData(stringList)
        pca = decomposition.PCA(n_components=2)

        #the var below is the actual compressed data
        principleComponents = pca.fit_transform(dfData)
        dfPC = pd.DataFrame(principleComponents)
        return dfPC
