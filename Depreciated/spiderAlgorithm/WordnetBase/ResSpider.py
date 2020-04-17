import pandas as pd
import json
import operator

from nltk.corpus import wordnet as wn

from nltk.corpus import wordnet_ic
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

#So this is pretty much just Spider but with Wordnet for similarity scoring
#This DOES NOT use Wordnet's query expansion/lemma abilities

#Uses Resnik Similarity 
#See here for good resource: http://www.nltk.org/howto/wordnet.html

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

class ResSpider:

    def __init__(self):
        #Load up external categories
        self.categoryNames = getCategories()
        self.categories = assembleCategories()
    
    #loop through all special terms of category list
    #return score
    def __compareCategory(self, text, category):
        scores = []
        
        for t in text.split():
            
            try:
                #Considers only the most common synset
                textSyn = wn.synsets(t)[0]
            
                for seedWord in category:
                    #format that you need
                
                    try:
                        seedSyn = wn.synsets(seedWord)[0]
                        sim = textSyn.res_similarity(seedSyn, semcor_ic)
                        if sim is not None: scores.append(sim)       

                    except:
                        continue
            except:
                continue

        return self.__score(scores)
    
    #I am not sure how we should pick the score, so rn it's just an average basically
    def __score(self, scores):
        total = 0
        for x in scores:
            total = total + x

        if len(scores) == 0: return 0
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


