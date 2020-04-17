import pandas as pd
from datamuse import datamuse
import json
import string

api = datamuse.Datamuse()


def generateSyns(cat):
    """Given a category, generate some synonyms for given terms.
    Also, add the words already in there as synonyms.

    cat = ONE category
    """
    result = []

    #Enter each keyword into the result in the proper format
    for keyword in cat:
        result = result + [{'word': keyword, 'score': -1, 'tags':[]}]

    #Now find all of syns

    for index, phrase in enumerate(cat):
        #40 was a sort of arbitrary choice
        if index <= 50:
            result = result + api.words(ml=phrase)

    return result

def generateAll(orig):
    """Run generateSyns on each and every category.
    Then, put all of it into a nicely formatted list of SETS
    orig = categories in the original format
    """

    #Formatted should be a dictionary of sets
    all_formatted = {}
    for key,value in orig.items():
        unformatted = generateSyns(value)

        #now generate formatted
        #It should be a set to prevent any duplicate entries
        one_formatted = set()
        for u in unformatted:
            one_formatted.add(u['word'].lower())

        all_formatted[key] = one_formatted

    return all_formatted


def calcInCategory(Txt, keywords):
    """Given category's keywords (as array of words) and some text Txt,
    calculate the degree to which TXT falls into that category.
    Defined as:
    (number of words in Txt that falls in category) / (total words in Txt)
    """
    #Must ensure that everything is a string type
    txt = str(Txt).lower()   #So capitalization doesn't screw this up

    #So you can use a set
    keywords = map(str.lower, keywords)

    inCategory = 0
    # totalTxt = len(txt)
    # counting the number of spaces would tell us the number of words CLD
    totalWords = 0

    #Select a word from txt and see if it's in keywords
    for word in txt.split():
        if word in keywords:
            inCategory = inCategory + 1
        totalWords += 1
    # Should totalTxt be the length of the entire Txt, or of the number of words

    return (inCategory/totalWords)



def catToColumns(df, scanned_col, categories):
    """Add columns for each category and set them to calcInCategory() result.
    If the text from scanned_col fits into a particular category, set the
    necessary column equal to True.

    df = original database
    scanned_col = string label for which column we want to analyze
    categories = original format of categories
    """

    if categories is None:
        with open('../misc/categories.json', 'r') as f:
            categories = json.load(f)

    form_c = generateAll(categories)    #formatted version of categories

    #Now loop through, generating a new column with each pass and
    #setting its value equal to calcInCategory() of scanned_col's value
    for key,c in form_c.items():
        df[key] = df[scanned_col].apply(calcInCategory, keywords=c)

    return df

