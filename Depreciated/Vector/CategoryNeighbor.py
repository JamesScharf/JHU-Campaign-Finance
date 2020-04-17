import numpy as np
import statistics
from collections import defaultdict, OrderedDict
from numpy.linalg import norm
import operator
import random
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
import json
from textblob import TextBlob
from datamuse import datamuse
from textblob import Word


stop_words = set(stopwords.words('english')) 
stemmer = SnowballStemmer("english")

def categorySet():
    '''
    Load up all of the categories into one big array
    '''
    categories = json.load(open("../vars/categories.json"))

    #loop through and make flat list of all
    catSet = set()
    for cat in categories:
        toAdd = []
        for term in categories[cat]:
            temp = term.upper()
            toAdd.extend(temp.split())
        catSet.update(toAdd)
    
    return catSet

api = datamuse.Datamuse()

def getWord(word):
    '''
    Datamuse
    '''
    wordList = api.words(ml = word)
    
    results = set()
    for wordDict in wordList:
        results.add(wordDict["word"].upper())
    return results

def synList(catSet):
    '''
    Generate a list of synonyms for every word in the catSet
    '''
    
    synSet = set()
    for word in catSet:
        synSet.update(set(getWord(word)))
    return synSet

catWords = categorySet()
synonyms = synList(catWords)

#remove stopwords and punctuation
#apply spell checking
def preprocess(phrase):
    #phrase is split word
    punct = "`~!@#$%^&*()_+=|\}]{[':;?/>.<,"
    nums = "1234567890"
    phrase = [w for w in phrase if not w in stop_words]
    phrase = [w for w in phrase if not w in punct]
    phrase = [w for w in phrase if not w in nums]
    #phrase = [stemmer.stem(w) for w in phrase]
    #tb = TextBlob(" ".join(phrase))
    #phrase = tb.correct().split()

    return phrase


class Document:
    def __init__(self, id, category, phrase):
        #phrase is a split-up word
        self.phrase = phrase
        self.category = category
        self.id = id
        #below are how many times a word appears
        self.freqs = self.count(phrase)
    
    def count(self, phrase):
        freqs = dict()
        for word in phrase:
            freqs[word] = phrase.count(word)
        return freqs

    def __repr__(self):
        return f"\nID: {self.id}\nPHRASE: {self.phrase}\n"


def process_tsv(fileName):
    '''
    This is built to process a file that only has
    purpose codes in them.
    '''
    documents = []

    with open(fileName) as f:
        count = 1
        for line in f:
            #if we did preprocessing
            #we should do it here

            splitUp = line.split(",")
            category = splitUp[len(splitUp) - 1]
            
            catLoc = line.index(category)

            phrase = ""
            count = 0
            
            while (count < catLoc - 1):
                phrase += line[count]
                count += 1

            sent = preprocess(word_tokenize(phrase))
            doc = Document(count, category, sent)
            documents.append(doc)
            
            #now add in the fake data
            #extras = createTrainingData(sent)
            #fakeDocs = []
            #for sentence in extras:
            #    fakeDocs.append(Document(count, category, sentence))

            #documents.extend(fakeDocs)
            count += 1

    return documents

def vectorize(doc, method, numDocs, freqs):
    '''
    Turn a sentence into a vector
    sentence = a split up string
    method = what weighting method we're trying
    numDocs = the total number of documents in our training set
    freqs = the frequencies of each word in the training corpus (how often each word appears)
    '''

    if method == "tf":
        return termFreq(doc)
    if method == "tfidf":
        return tfidf(doc, numDocs, freqs)
    if method == "tfCat":
        return tfCat(doc)
    if method == "tfidfCat":
        return tfidfCat(doc, numDocs, freqs)


def termFreq(doc):
    '''
    Term-frequency weighting.
    Pretty much just a count.
    '''
    return doc.freqs
     

def tfidf(doc, numDocs, freqs):
    tf = termFreq(doc)
    for key, value in tf.items():
        if key in freqs:
            df_t = freqs[key]
        else:
            df_t = 0
        if df_t != 0:
            tf[key] = tf[key] * np.log(numDocs / df_t)
        else:
            tf[key] = 0
    
    return tf

def tfCat(doc):
    tf = termFreq(doc)
    for key, value in tf.items():
        if key in catWords:
            tf[key] = tf[key] * 50
        elif key in synonyms:
            tf[key] = tf[key] * 20
    return tf

def tfidfCat(doc, numDocs, freqs):
    tf = termFreq(doc)
    for key, value in tf.items():
        if key in freqs:
            df_t = freqs[key]
        else:
            df_t = 0
        if df_t != 0:
            tf[key] = tf[key] * np.log(numDocs / df_t)
        else:
            tf[key] = 0

        if key in catWords:
            tf[key] = tf[key] * 50
        elif key in synonyms:
            tf[key] = tf[key] * 20
        
    
    return tf


def dictdot(x, y):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y):
    #IMPLEMENTED
    #numerator is num
    num = 2 * dictdot(x, y)
    if num == 0:
        return 0

    #total number of terms
    #t = len(x) + len(y)
    return num / (sum(list(x.values())) + sum(list(y.values())))

def jaccard_sim(x, y):
    
    #sums

    sum_max = 0.0
    sum_min = 0.0
    
    #make union of the keys in both
    x_keys = set(x.keys())
    y_keys = set(y.keys())
    comboKeys = x_keys.union(y_keys)
    
    #loop through keys now
    for k in comboKeys:
        #try to get x[k]'s value, but make it 0 if it doesn't exist in dict
        xWeight = x.get(k, 0)
        yWeight = y.get(k, 0)

        #min
        sum_min = sum_min + min(xWeight, yWeight)
        sum_max = sum_max + max(xWeight, yWeight)

    
    if sum_max == 0:
        return 0

    return sum_min/sum_max

def overlap_sim(x, y):

    num = dictdot(x, y)
    if num == 0:
        return 0

    return num / min(sum(list(x.values())), sum(list(y.values())))



def getFreqs(corpus):
    '''
    Calculate frequencies of words
    All across the corpus
    '''
    bigStringList = []
    for doc in corpus:
        bigStringList += doc.phrase
    
    #get frequency of all words in corpus
    freqs = dict()
    for word in bigStringList:
        freqs[word] = bigStringList.count(word)
    return freqs

def vectorizeCorpus(documents, freqs, method):
    if method == "tfidf" or method == "tfidfCat":
        freqs = getFreqs(documents)
    else:
        freqs = None

    numDocs = len(documents)
    for d in documents:
        d.vectorized = vectorize(d, method, numDocs, freqs)
    
    return documents

def relevanceScores(query, vectorized, freqs, rel_method, weightMethod):
    '''
    Given a query (normal string) and an
    array of vectorized documents (documents with .vectorized attribute),
    return an ordered dictionary of the most relevant documents
    '''
    relevanceScores = OrderedDict()

    query = query.upper()
    vecDoc = Document(0, -1, preprocess(word_tokenize(query)))
    
    numDocs = len(vectorized)
    vectorQuery = vectorize(vecDoc, weightMethod, numDocs, freqs)

    for d in vectorized:
        if rel_method == "cosine":
            score = cosine_sim(vectorQuery, d.vectorized)
        if rel_method == "jaccard":
            score = jaccard_sim(vectorQuery, d.vectorized)
        if rel_method == "dice":
            score = dice_sim(vectorQuery, d.vectorized)
        if rel_method == "overlap":
            score = overlap_sim(vectorQuery, d.vectorized)
        
        relevanceScores[d] = score
    rank = relevanceScores

    return sorted(rank.items(), key=operator.itemgetter(1), reverse = True)


def categorize(k, query, vectorized, freqs, rel_method, weightMethod):
    '''
    Performs a K Nearest Neighbor classifier

    query = the purpose code that we want to categorize
    k = the number of nearby documents that we must get the category of
        (so the category label is equal to the most common category in the nearest K documents)
        Must be an odd number or else there will be ties
    '''
    '''
    syns = set()
    splitSentence = query.split()
    for word in splitSentence:
        w = Word(word)
        syns.add(w.pluralize().upper().replace("_", " "))
        syns.add(w.singularize().upper().replace("_", " "))

        
        #use synsets to find synonyms
        wordSyns = Word(word).get_synsets()
        tempWords = set()
        if len(wordSyns) > 3:
            wordSyns = wordSyns[0:3]
        for syn in wordSyns:
            tempWords.update(set(syn.lemma_names()))

        tempTemp = tempWords.copy()
        finalCopy = set()
        for tempWord in tempTemp:
            finalCopy.add(tempWord.upper().replace("_", " "))
            tempWords.add(str(Word(tempWord).pluralize()).upper().replace("_", " "))
            tempWords.add(str(Word(tempWord).singularize()).upper().replace("_", " "))

        syns.update(set(finalCopy))
        

    expanded = splitSentence + list(syns)
    query = " ".join(expanded)
    '''
    scores = relevanceScores(query, vectorized, freqs, rel_method, weightMethod)

    count = 0
    categories = []
    while (count < k):
        categories.append(scores[count][0].category)
        count += 1

    counts = Counter(categories)
    return counts.most_common(1)[0][0] 
    

def test(k, weightMethod, rel_method):

    documents = process_tsv("C:/Users/schar/Documents/PoliSciResearch/vars/purposeCodeTraining-train.tsv")
    freqs = getFreqs(documents)

    trainDocuments = vectorizeCorpus(documents, freqs, weightMethod)

    testDocuments = process_tsv("C:/Users/schar/Documents/PoliSciResearch/vars/purposeCodeTraining-dev.tsv")
    testDocuments = vectorizeCorpus(testDocuments, freqs, weightMethod)


    file = open("KNN_Errors.txt", "w")

    totalCorrect = 0.0
    totalTests = len(testDocuments)
    for doc in testDocuments:
        testPhrase = " ".join(doc.phrase)
        correctCat = doc.category
        result = categorize(k, testPhrase, trainDocuments, freqs, rel_method, weightMethod)

        file.write(f"\n {testPhrase}\n")
        file.write(f"CORRECT: {correctCat}\n")
        file.write(f"GUESS: {result} \n\n")

        if correctCat in result:
            totalCorrect += 1
    
    score = str(totalCorrect / totalTests)
    file.write(f"TOTAL ACCURACY: {score}\n")
    return score

def experiment():

    k = 1
    while (k < 55):
        '''
        print("K      SCORE        WEIGHT METHOD        SIMILARITY FUNCTION")
        weightFunc = "tf"
        simFunc = "cosine"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tf"
        simFunc = "jaccard"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tf"
        simFunc = "dice"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tf"
        simFunc = "overlap"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")

        print("K      SCORE        WEIGHT METHOD        SIMILARITY FUNCTION")
        weightFunc = "tfidf"
        simFunc = "cosine"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfidf"
        simFunc = "jaccard"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfidf"
        simFunc = "dice"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfidf"
        simFunc = "overlap"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        print("K      SCORE        WEIGHT METHOD        SIMILARITY FUNCTION")
        weightFunc = "tfCat"
        simFunc = "cosine"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfCat"
        simFunc = "jaccard"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfCat"
        simFunc = "dice"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfCat"
        simFunc = "overlap"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        '''
        print("K      SCORE        WEIGHT METHOD        SIMILARITY FUNCTION")
        weightFunc = "tfidfCat"
        simFunc = "cosine"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfidfCat"
        simFunc = "jaccard"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfidfCat"
        simFunc = "dice"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")
        weightFunc = "tfidfCat"
        simFunc = "overlap"
        responses = test(k, weightFunc, simFunc)
        print(f"{k}     {responses}         {weightFunc}        {simFunc}")

        k += 2

documents = process_tsv("C:/Users/schar/Documents/PoliSciResearch/vars/purposeCodeTraining-train.tsv")
freqs = getFreqs(documents)

trainDocuments = vectorizeCorpus(documents, freqs, "tfidfCat")

testDocuments = process_tsv("C:/Users/schar/Documents/PoliSciResearch/vars/purposeCodeTraining-dev.tsv")
testDocuments = vectorizeCorpus(testDocuments, freqs, "tfidfCat")

def categorizePhrase(phrase):
    return categorize(15, phrase, trainDocuments, freqs, "jaccard", "tfidfCat")