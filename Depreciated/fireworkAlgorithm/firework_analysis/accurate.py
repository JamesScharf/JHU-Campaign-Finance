import pandas as pd
import firework
import numpy as np

class AlgorithmAnalyzer:

    def __init__ (self, fileName):
        # reading survey from fileName
        survey = pd.read_csv(fileName)
        purposeCodes = survey['Purpose Code']
        answers = survey['Final Estimate']
        self.purposeCodes = purposeCodes.values
        self.answers = answers.values
    
    def getScore(self, storeIncorrect):
        # getting accuracy score, and recording incorrect comparisions
        f = firework.FireworkAlgorithm(useSaved = True)
        correct = 0
        total = 0
        incorrectPC = []
        incorrectLabelings = []
        correctLabelings = []
        for x in range (self.purposeCodes.size):
            fireworkGuess = f.categorizePhrase(self.purposeCodes[x])
            if (fireworkGuess.lower() == self.answers[x].lower()):
                correct += 1
            else:
                incorrectPC.append(self.purposeCodes[x])
                incorrectLabelings.append(fireworkGuess)
                correctLabelings.append(self.answers[x])
            total += 1

            # printing progress bar
            if (x % (int(self.purposeCodes.size / 10)) == 0): # if we are multiple of 10% done
                print(str(round(100*round(x / self.purposeCodes.size, 2), 1)) + " percent complete") # print it
        print()

        # store incorrect outputs
        with open(storeIncorrect, 'w') as file:
            file.write("Purpose Code, Correct, Guess")
            file.write("\n")
            for x in range (len(incorrectPC)):
                file.write(incorrectPC[x] + ", " + correctLabelings[x] + ", " + incorrectLabelings[x])
                file.write("\n")
            file.close()
        return round(correct / total, 3)
            

a = AlgorithmAnalyzer("purposecode_labeled_first100.csv")
print(a.getScore("errors.csv"))