from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.corpus import reuters


#Good explanation of TfidfVectorizer here: https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms
#Or here: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
from sklearn.feature_extraction.text import TfidfVectorizer


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



#Expand wordnet tools
class WordExpander():
    
    def __init__(self):
        self.ps = PorterStemmer()

    def stemmedSyns(self, word):
        wordSyns = wn.synsets(word)

        return self.synsToStems(wordSyns)

    def stemmedMeronyms(self, word):
        wordSyns = wn.synsets(word)
        total = []
        try:
            meronyms = wordSyns[0].substance_meronyms()
            meronyms = meronyms + wordSyns[0].part_meronyms()
        
            return self.synsToStems(meronyms)
        except:
            return []

    def stemmedHolonyms(self, word):
        wordSyns = wn.synsets(word)
        total = []
        try:
            holonyms = wordSyns[0].substance_holonyms()
            holonyms = wordSyns[0].part_holonyms()

            return self.synsToStems(holonyms)
        except:
            return []

    def synsToStems(self, syns):
        total = []
        for x in syns:
            lems = x.lemmas()
            for lem in lems:
                total.append(self.ps.stem(lem.name().replace("_", " ")))


        return list(set(total))
    
    #returns a frequency array correlating to the inputted array
    #ex. plug in: ["word1", "word2", "word3", "word4"] and "word1 word1 x y z word3"
    #   and return: [2, 0, 0, 1, 0]
    def freq(self, wordList, text):
        freq = []

        text = text.lower()

        for wl in wordList:
            freq.append(text.count(wl))

        return freq
    

    def loadCorpus(self):
        #the corpus as one big string
        self.corpus = reuters.words()
        self.tVectorizer = TfidfVectorizer()
        #build the vocabulary from the corpus
        self.tVectorizer.fit(self.corpus)

    #Get a frequency using all words in the reuters corpus
    #must run loadCorpus() first
    def freqCorpus(self, text):
        text = [text]
        data = self.tVectorizer.transform(text)
        return data.toarray().flatten().tolist()
        
             

   #Expand a phrase i.e. take every word and get its synonyms
   #args are boolean
    def expand(self, text, syns, meronyms, holonyms):
        result = text.split()
        temp = text.split()

        for t in temp:
            if syns is True:
                result.extend(list(self.stemmedSyns(t)))

            if meronyms is True:
                result.extend(list(self.stemmedMeronyms(t)))

            if holonyms is True:
                result.extend(list(self.stemmedHolonyms(t)))
            result.append(t)

        return set(result)


    #Take the seed words and expand
    def expandSeeds(self, catSeeds):
        result = []
        for seedWord in catSeeds:
            result.extend(self.expand(catSeeds))

        return result

    #expand all categories
    def expandAllCategories(self):
        categories = assembleCategories()
        result = []

        for c in categories:
            result.append(self.expandSeeds(c))
        
        self.expandedCategories = result

        return result
        
