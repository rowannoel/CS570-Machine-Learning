"""
Name: Rowan Noel-Rickert
Naive Bayes implementation (categorical features only)
"""
import math
from typing import Dict, List, Any, Set
from collections import defaultdict

class NaiveBays:
    def __init__(self, trainData, targetAttribute: str, use_Laplace=True):
        self.targetAttribute = targetAttribute
        self.targetValues = list(trainData.discreteValues.get(targetAttribute, set()))
        if not self.targetValues:
            self.targetValues = list(set(trainData.featureData[targetAttribute]))

        #Get all the descriptive features except for the target
        self.features = [attr for attr in trainData.attributes if attr != targetAttribute]

        #Store possible values for each feature
        self.featureValues = {}
        for feature in self.features:
            if feature in trainData.discreteValues:
                self.featureValues[feature] = list(trainData.discreteValues[feature])
            else:
                self.featureValues[feature] = list(set(trainData.featureData[feature]))

        #If using laplace smoothing
        self.useLaplace = use_Laplace

        #Storing previous probabilities
        self.previous = {}

        """
        3D array for conditional probabilities 
        1st dimension: target values (class)
        2nd dimension: feature index
        3rd dimension: feature value
        """
        self.conditionalProbabilities = {}

        #train the model
        self._train(trainData)

    #train the naive bayes model on the data
    def _train(self, trainData):
        totalInstance = len(trainData.featureData[self.targetAttribute])

        #Count class occurances
        classCounts = {}
        for targetValue in trainData.featureData[self.targetAttribute]:
            if targetValue not in classCounts:
                classCounts[targetValue] = 0
            classCounts[targetValue] += 1

        #calculate previous probabilities
        for targetValue in self.targetValues:
            count = classCounts.get(targetValue, 0)
            #if using laplace smoothing do so now
            if self.useLaplace:
                self.previous[targetValue] = (count + 1)/(totalInstance + len(self.targetValues))
            else:
                self.previous[targetValue] = count/totalInstance if totalInstance > 0 else 0

        #initilize conditional probabilities structure
        for targetValue in self.targetValues:
            self.conditionalProbabilities[targetValue] = {}
            for feature in self.features:
                self.conditionalProbabilities[targetValue][feature] = {}

        #calculate the conditional probabilities for each feature value in each class
        for feature in self.features:
            featureValue = self.featureValues[feature]

            #for each class
            for targetValue in self.targetValues:
                #count how many times each feature value is in the class
                valueCounts = defaultdict(int)
                classTotal = 0

                #count the times
                for i in range(totalInstance):
                    if trainData.featureData[self.targetAttribute][i] == targetValue:
                        featValue = trainData.featureData[feature][i]
                        if featValue is not None:
                            valueCounts[featValue] += 1
                            classTotal += 1

                #calculate the probabilities for the feature values
                for featValue in featureValue:
                    count = valueCounts.get(featValue, 0)

                    #if using laplace smoothing
                    if self.useLaplace:
                        prob = (count + 1) / (classTotal + len(featureValue))
                    else:
                        prob = count / classTotal if classTotal > 0 else 0

                    #store it in our 3d structure
                    self.conditionalProbabilities[targetValue][feature][featValue] = prob

    #predict the class for a given instance
    def predict(self, instance):
        bestClass = None
        bestProb = float('-inf')

        #calculate probability for each class
        for targetValue in self.targetValues:
            #start with the log of previous probabilities
            logProb = math.log(self.previous[targetValue])

            #add log of conditional probabilities for each feature
            for feature in self.features:
                featValue = instance.get(feature)

                #skip if feature is none
                if featValue is None:
                    continue

                #get the probability of this feature value given this class
                if featValue in self.conditionalProbabilities[targetValue][feature]:
                    prob = self.conditionalProbabilities[targetValue][feature][featValue]
                else:
                    #handle unseen values
                    if self.useLaplace:
                        #with smoothing, they get a small probability
                        prob = 1 / (len(self.featureValues[feature])+1)
                    else:
                        #w/out laplace, this probability is 0
                        prob = 0

                #prevent overflow from very small probabilities
                if prob > 0:
                    logProb += math.log(prob)
                #if prob is 0 so the whole thing will be 0, using log use a very large neg number
                else:
                    logProb = float('-inf')
                    break

            #keep track of highest probability
            if logProb > bestProb:
                bestProb = logProb
                bestClass = targetValue

        return bestClass

    #print the model parameters
    def printModel(self):
        print("\n=== Naive bayes Model ===")

        print("\nPrior Probabilities:")
        for targetValue, prob in self.previous.items():
            print(f"  P({targetValue}) = {prob:.4f}")

        print("\nConditional Probabilities:")
        for feature in self.features:
            print(f"\nfeature: {feature}")
            for targetValue in self.targetValues:
                print(f"  Class: {targetValue}")
                for featValue in sorted(self.featureValues[feature]):
                    prob = self.conditionalProbabilities[targetValue][feature].get(featValue, 0)
                    print(f"  P({featValue}|{targetValue}) = {prob:.4f}")