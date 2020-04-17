
#Created by James A. Scharf on 2/1/19


#TO RUN: make new object and run obj.remove()

#Goal of this file is to create a linter of sorts
#i.e. remove unneeded words from the purpose codes
#or the words that throw everything off
#note that the primary source of stopwords is here: https://kb.yoast.com/kb/list-stops

class preprocessor:
    
    def __init__(self):
        self.punct = self.load_punct()
        self.wordsToRemove = self.load_words()

    #@return list of words to remove
    def load_words(self):
        fi = open("../vars/RemovedWords.txt", "r")
        toRemove = fi.read()
        return set(toRemove.split())
    
    #remove punctuation
    #must be done separately because spaces separate words
    def load_punct(self):
        fi = open("../vars/RemovedPunctuation.txt", "r")
        toRemove = fi.read()
        return set(toRemove.split())

    #Remove all of the unneeded punctuation and stopwords
    def remove(self, inpu):
        inp = " " + inpu.lower()

        for p in self.punct:
            inp = inp.replace(p, "")
        
    
        #split by whitespace
        #set because we can remove duplicate 
        #Need this one for removal
        for x in self.wordsToRemove:
            if (" " + x + " ") in inp:
                inp = inp.replace(x, "")

                
        return inp



#test
#pre = preprocessor()

