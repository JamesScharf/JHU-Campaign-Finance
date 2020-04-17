#This program exists to provide easy analysis of a data set

import spider as spider
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None #default = 'warn'

#purp['category'] = purp['pur'].apply(s.maxCategory, aggregation = None)

# Apply function to category to create new column (using spider)
# @param df = the dataframe to add new columns to (and return)
# @param col = a dataframe column (string) on which to run the .apply (like purpose)
# @param categories looks like: ["media", "digital", "administrative"...]
# @return df = a modified dataframe with category scores appended
def spiderApplyAggregateAll(spid, df, col, categories):

    for c in categories:
        df[c] = df[col].astype(str).apply(spid.categoryAggregate, category = c)

    return df


#Get a list of all candidates in the set
# @param df = the dataframe holding the data
# @param col = the column that has the names of the candidates
#
#NOTE!!!! DOES NOT TAKE SPELLING MISTAKES INTO ACCOUNT
def cand_list(df, col):
    return set(df[col])


# Returns a summary of the candidate as a dictionary
# Note that it will double count overlapping values
# @param df = something like purpose codes
def summary(spid, df, cand_nam):
    #PROBLEM!!! THIS ONLY MATCHES EXACT NAMES
    df = df.dropna()
    candidateData = df[(df['can_nam'].str.contains(cand_nam))]

    candidateData['category'] = candidateData['pur'].astype(str).apply(spid.maxCategory, aggregation = None)

    summary = dict()

    summary["Name"] = cand_nam
    summary["Num. Media Occurences"] = (candidateData['category'].astype(str).contains("media")).sum()
    summary["Num. Digital Occurences"] = (candidateData['category'].astype(str).contains("digital")).sum()
    summary["Num. Polling Occurences"] = (candidateData['category'].astype(str).contains("polling")).sum()
    summary["Num. Legal Occurences"] = (candidateData['category'].astype(str).contains("legal")).sum()
    summary["Num. Field Occurences"] = (candidateData['category'].astype(str).contains("field")).sum()
    summary["Num. Consulting Occurences"] = (candidateData['category'].astype(str).contains("consulting")).sum()
    summary["Num. Administrative Occurences"] = (candidateData['category'].astype(str).contains("administrative")).sum()
    summary["Num. Fundraising Occurences"] = (candidateData['category'].astype(str).contains("fundraising")).sum()

    summary["Num. Misc. Occurences"] = (candidateData['category'] == "misc").sum()

    media = (candidateData['category'].astype(str).contains("media"))
    digital = (candidateData['category'].astype(str).contains("digital"))
    polling = (candidateData['category'].astype(str).contains("polling"))
    legal = (candidateData['category'].astype(str).contains("legal"))
    field = (candidateData['category'].astype(str).contains("field"))
    consulting = (candidateData['category'].astype(str).contains("consulting"))
    administrative = (candidateData['category'].astype(str).contains("administrative"))
    fundraising = (candidateData['category'].astype(str).contains("fundraising"))

    misc = candidateData[candidateData['category'] == 'misc']

    #Generate costs
    medCost = media['exp_amo'].sum()
    digCost = digital['exp_amo'].sum()
    polCost = polling['exp_amo'].sum()
    legCost = legal['exp_amo'].sum()
    fieCost = field['exp_amo'].sum()
    conCost = consulting['exp_amo'].sum()
    admCost = administrative['exp_amo'].sum()
    misCost = misc['exp_amo'].sum()
    funCost = fundraising['exp_amo'].sum()

    summary["Media Costs"] = medCost
    summary["Digital Costs"] = digCost
    summary["Polling Costs"] = polCost
    summary["Legal Costs"] = legCost
    summary["Field Costs"] = fieCost
    summary["Consulting Costs"] = conCost
    summary["Administrative Costs"] = admCost
    summary["Misc. Costs"] = misCost
    summary['Fundraising Costs'] = funCost

    #DO SOME GEOGRAPHICAL DATA

    #Cost graphs
    temp = [medCost, digCost, polCost, legCost, fieCost, conCost, admCost, funCost, misCost]
    titles = ["Media", "Digital", "Polling", "Legal", "Field", "Consulting", "Administrative", "Fundraising", "Misc"]
    costs = pd.Series(data = temp, index = titles)

    ax = costs.plot(kind = "bar")
    ax.set_title("Dollars Per Category for " + cand_nam)
    ax.set_ylabel("Dollars Spent (Scaled)")
    ax.set_xlabel("Categories")

    #Make a radar graph of the $ spread per category




    return summary


#Make a summary of the year
#   @param df = the purpose codes dataframe
#   @param year = ex. 2010 (integer)
def sumYear(df, year):
    election = df[df['election'] == year]

    media = (election['category'].dropna().astype(str).contains("media"))
    digital = (election['category'].dropna().astype(str).contains("digital"))
    polling = (election['category'].dropna().astype(str).contains("polling"))
    legal = (election['category'].dropna().astype(str).contains("legal"))
    field = (election['category'].dropna().astype(str).contains("field"))
    consulting = (election['category'].dropna().astype(str).contains("consulting"))
    administrative = (election['category'].dropna().astype(str).contains("administrative"))
    fundraising = (election['category'].dropna().astype(str).contains("fundraising"))
    #Now generate sums
    #Variable naming scheme: first three letters of category + 'Cost' + Year
    medCost = media['exp_amo'].sum()
    digCost = digital['exp_amo'].sum()
    polCost = polling['exp_amo'].sum()
    legCost = legal['exp_amo'].sum()
    fieCost = field['exp_amo'].sum()
    conCost = consulting['exp_amo'].sum()
    admCost = administrative['exp_amo'].sum()
    funCost = fundraising['exp_amo'].sum()

    misCost = misc['exp_amo'].sum()

    temp = [medCost, digCost, polCost, legCost, fieCost, conCost, admCost, funCost, misCost]
    titles = ["Media", "Digital", "Polling", "Legal", "Field", "Consulting", "Administrative", "Fundraising", "Misc"]
    costs = pd.Series(data = temp, index = titles)

    ax = costs.plot(kind = "bar")
    ax.set_title("Dollars Per Category in " + str(year))
    ax.set_ylabel("Dollars Spent (hundred million) Across Categories")
    ax.set_xlabel("Categories")
    print("Note that there is double counting of entries that fall in two categories")


