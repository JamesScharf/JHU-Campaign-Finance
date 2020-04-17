import wordscorer as ws

# defining the purpose phrase
purposePhrase = "Transportation between social events"

# ws offers an ability to grab the categories as an array
categories = ws.getCategories()

# pre-setting the final score to be the right size
categoryScores = [None] * len(categories)

# generating all the realted words to our passed purpose phrase
# output is String[]
# this method uses the api
purposeWords = ws.generateRelatedWords(purposePhrase)

# iterating through all the categories
for i in range(len(categories)): 
    # generate all the words that are related to the categories
    # include_synonyms also adds the synonyms available in the categories.json file
    categoryWords = ws.getRelatedCategoryWords(categories[i], include_synonyms = True)
    
    # ws.scoreSimilarity looks at percentage of words that are common
    # returns a float of similarity
    categoryScores[i] = ws.scoreSimilarity(categoryWords, purposeWords)

print(categories)
print(categoryScores)