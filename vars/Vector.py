import numpy as np
import statistics
from collections import defaultdict, OrderedDict


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


def process_csv(fileName):
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
            doc = Document(count, -1, line.split())
            documents.append(doc)
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
    if method == "tfidf":
        freqs = getFreqs(documents)
    else:
        freqs = None

    numDocs = len(documents)
    for d in documents:
        d.vectorized = vectorize(d, method, numDocs, freqs)
    
    return documents

def rankCorpus(query, vectorized, freqs, rel_method, weightMethod):
    '''
    Given a query (normal string) and an
    array of vectorized documents (documents with .vectorized attribute),
    return an ordered dictionary of the most relevant documents
    '''
    relevanceScores = OrderedDict()
    query = query.upper()
    vecDoc = Document(0, -1, query)
    
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
    
    return relevanceScores


documents = process_csv("C:/Users/schar/Documents/PoliSciResearch/vars/barePurposeCodes.csv")
freqs = getFreqs(documents)
documents = vectorizeCorpus(documents, freqs, "tfidf")

print(rankCorpus("digital media", documents, freqs, "cosine", "tfidf"))
