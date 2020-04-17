import pandas as pd
from datamuse import datamuse
import json
import operator
import preprocessor as pp

api = datamuse.Datamuse()


#Up until SpiderAlgorithm, these are mostly the same functions as in wordscorer.py

# reading the categories into "categories"
def getCategories():
    #'/home/jscharf/Research/polisciresearch/misc/categories.txt'
    with open ('../vars/categories.txt', 'r') as f:
        categories = f.read().split()
    return categories


# read categories synonyms into "categoriesRaw"
# @param String category - the category to search the json file for
def getCategorySynonyms(category):
    #'/home/jscharf/Research/polisciresearch/misc/categories.json'
    with open('../vars/categories.json', 'r') as f:
        categoriesRaw = json.load(f)
    # assuring that the passed category is an actual category
    try:
        return categoriesRaw[category]
    except:
        return [category]


# Remove meaningless words from an array of strings
# @param phrase ex. looks like ["this", "is", "my", phrase"]
# @return the cleaned version

# takes two arrays and calculates the number of similar words
# @param String[] wordArray1 - an array of words
# @param String[] wordArray2 - another array of words
def scoreSimilarity(wordArray1, wordArray2):
    similar = 0
    for word in wordArray1:
        if (word in wordArray2):
            similar += 1
    return (round(similar / len(wordArray1), 3))


# The functions that follow try to implement the new algorithm

class SpiderAlgorithm:
    'Implements new alg which takes scores into account'

    def __init__(self):
        self.prepro = pp.preprocessor() 
        #Categories without synonyms
        self.catsLabelsOnly = getCategories()
        #Categories with synonyms
        self.catsAndSyns = self.catsWithSynonyms()
        #The following function creates a variable called
        #disjointed which is a dictionary of terms and their
        #disjointed categories
        self.dijointAll()
        self.prepro = pp.preprocessor()

    # Load all categories as dictionaries with their synonyms
    def catsWithSynonyms(self):
        cWithSyns = dict();     #The dictionary storing synonyms
        #Loop through each and append
        for c in self.catsLabelsOnly:
            cWithSyns[c] = getCategorySynonyms(c)

        self.sCategories = cWithSyns

        #There is this very strange issue where
        return cWithSyns

    #This one is just a wrapper on the api call
    # @param word something like "media"
    def getWordAndScoreFromAPI(self, word):
        wordDict = api.words(ml = word)
        return wordDict


    # Run getWordAndScoreFromAPI on all
    # of the synonyms and root word of the category
    # @param category = one category's keywords/synonyms in format like:
    # "['media', 'TV', 'television']"
    def catWordsAndScores(self, category):

        allWordsAndScores = []

        count = 0
        for c in category:
            if (count > 40):
                break

            #Add the word itself to it
            #Give it some crazy high score since it's important
            allWordsAndScores.extend([{'word': c, "score": 9999999, 'tags':[]}])
            allWordsAndScores.extend(self.getWordAndScoreFromAPI(c))
            count = count + 1

        return allWordsAndScores

    # Use when comparing the same word if it's found in two different categories
    # And return which one is greater
    # @param word1 looks like {'word': 'someWord', 'score': 100}
    # @param word2 same as word1
    def compareScores(self, word1, word2):
        if "score" in word1 and "score" in word2:
            if word1["score"] > word2["score"]:
                return 1
            elif word1["score"] == word2["score"]:
                return 0
            else:
                return -1
        else:
            return -1

    # Given two categories, find what words they share
    # and remove those from the category that it is less similar to
    # @param cat1 A category in format from catWordsAndScores
    # @param cat2 Same as above
    def makeDisjoint(self, cat1, cat2):

        for c1 in cat1:
            for c2 in cat2:
                #NOTE!!!!!!
                # I can't decide whether the next phrase should use
                # "in" or ==
                # It appears that "in" reduces overlap,
                # but this doesn't make much senes.
                # I think this works because it removes stop words
                # and smaller, less specific phrases.
                # So For now, it says "in"
                if (c1["word"] in c2["word"]):
                    if self.compareScores(c1, c2) == 1:
                        #Score of c1 is greater than c2
                        #So the word does not belong in c2
                        cat2.remove(c2)
                    elif self.compareScores(c1, c2) == -1:
                        if c1 in cat1: cat1.remove(c1)

        return [cat1, cat2]

    #Remove all stopwords and more from categories
    def preprocess(self, text):
        #processed = self.prepro.remove(text)
        #return processed
        return text
        

    # Apply to all of the categories
    # Return disjointed versions in array

    # Remove duplicates in one category
    # Where category is some string "media", "polling"...
    # PROBLEM: Just accepts the first occurance of a word if there are duplicates
    def removeDuplicates(self, category):
        alreadyEncountered = []

        #Contains all of the words and scores now
        cat = self.disjointed[category]

        for c in cat:
            if (c["word"] in alreadyEncountered):
                cat.remove(c)
            else:
                alreadyEncountered.append(c["word"])

        self.disjointed[category] = cat
        return cat

    # Make all categories disjoint
    # I intentionally do not loop through every category
    # because some would probably not insersect.
    # So, I didn't want to waste computing power.
    def dijointAll(self):
        catsWithSyns = self.catsWithSynonyms()
        #media, digital, polling
        #legal, field, consulting, administrative

        #I only apply disjoint all on categories that I know should have
        #significant overlap. It would probably be a waste of computing power
        #to compare every category with each other.

        media = self.catWordsAndScores(catsWithSyns["media"])
        digital = self.catWordsAndScores(catsWithSyns["digital"])
        polling = self.catWordsAndScores(catsWithSyns["polling"])
        legal = self.catWordsAndScores(catsWithSyns["legal"])
        field = self.catWordsAndScores(catsWithSyns["field"])
        consulting = self.catWordsAndScores(catsWithSyns["consulting"])
        administrative = self.catWordsAndScores(catsWithSyns["administrative"])
        fundraising = self.catWordsAndScores(catsWithSyns["fundraising"])

        temp = self.makeDisjoint(media, digital)
        media = temp[0]
        digital = temp[1]

        temp = self.makeDisjoint(media, polling)
        media = temp[0]
        polling = temp[1]

        temp =  self.makeDisjoint(digital, consulting)
        digital = temp[0]
        consulting = temp[1]

        temp =  self.makeDisjoint(legal, field)
        legal = temp[0]
        field = temp[1]

        temp =  self.makeDisjoint(legal, consulting)
        legal = temp[0]
        consulting = temp[1]

        temp =  self.makeDisjoint(media, administrative)
        media = temp[0]
        administrative = temp[1]

        temp =  self.makeDisjoint(digital, administrative)
        digital = temp[0]
        administrative = temp[1]

        temp =  self.makeDisjoint(field, administrative)
        field = temp[0]
        administrative = temp[1]

        temp =  self.makeDisjoint(consulting, administrative)
        consulting = temp[0]
        administrative = temp[1]

        temp =  self.makeDisjoint(fundraising, administrative)
        fundraising = temp[0]
        administrative = temp[1]

        temp =  self.makeDisjoint(fundraising, consulting)
        fundraising = temp[0]
        consulting = temp[1]

        self.disjointed = dict()
        self.disjointed["media"] = media
        self.disjointed["digital"] = digital
        self.disjointed["polling"] = polling
        self.disjointed["legal"] = legal
        self.disjointed["field"] = field
        self.disjointed["consulting"] = consulting
        self.disjointed["administrative"] = administrative
        self.disjointed["fundraising"] = fundraising


        self.removeDuplicates("media")
        self.removeDuplicates("digital")
        self.removeDuplicates("polling")
        self.removeDuplicates("legal")
        self.removeDuplicates("field")
        self.removeDuplicates("consulting")
        self.removeDuplicates("administrative")
        self.removeDuplicates("fundraising")

        return self.disjointed

    # Get the score of a phrase in one category
    # (gets an aggregate of all similarity index scores)
    # @param phrase = "some phrase like this"
    # @param category = ex. "media", "polling", ...
    # @returns Aggregate score
    def categoryAggregate(self, phrase, category):
        
        
        phrase = phrase.lower()
        #remove stop words and the like
        phrase = self.preprocess(phrase)
        cat = self.disjointed[category]
        aggregate = 0

        #Loop through the category
        for c in cat:
            if c["word"] in phrase:
                #Add to aggregate
                if "score" in c:
                    aggregate = aggregate + c["score"]

        return aggregate

    # Gets the score of the phrase in each category.
    # ***This is the function that you would use with Apply in Pandas
    # @param phrase = "some phrase like this"
    # @returns a dictionary of scores
    def aggregateAll(self, phrase):
        result = dict()
        result["media"] = self.categoryAggregate(phrase, "media")
        result["digital"] = self.categoryAggregate(phrase, "digital")
        result["polling"] = self.categoryAggregate(phrase, "polling")
        result["legal"] = self.categoryAggregate(phrase, "legal")
        result["field"] = self.categoryAggregate(phrase, "field")
        result["consulting"] = self.categoryAggregate(phrase, "consulting")
        result["administrative"] = self.categoryAggregate(phrase, "administrative")
        result["fundraising"] = self.categoryAggregate(phrase, "fundraising")


        return result

    # Get the highest category
    # @param aggregation = the result of aggregateAll(phrase)
    # @param phrase = "some phrase like this"
    def maxCategory(self, phrase, aggregation):
        if aggregation is None:
            aggregation = self.aggregateAll(phrase)

        #I took the following line from:
        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary


        for x in aggregation:
            #Checks whether there is any significant overlap between categories
            if (aggregation[x] != 0):
                largest = max(aggregation, key=aggregation.get)
                valOfLargest = aggregation[largest]

                #Now remove that from dictionary
                del aggregation[largest]

                #And find the second largest
                secondLargest = max(aggregation, key=aggregation.get)
                valSecond = aggregation[secondLargest]

                # commented out for better accuracy testing. 
                # goal is binary "yes" or "no" results
                '''#If they're almost the same values, then return a combined value
                if (valOfLargest != 0) and (valSecond != 0):
                    if ((valSecond/valOfLargest) > 0.98):
                        return largest + "+" + secondLargest'''
                return largest

        # Return if it matches none of the categories
        return "misc"

    #Just wrapper for accurate.py
    def categorizePhrase(self, phrase):
        return self.maxCategory(phrase, None)

