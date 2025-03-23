"""
Name: Rowan Noel-Rickert

Write an implementation of Naive Bayes.  You can assume that the target feature and all input
features are categorical.  Have a main that trains a model on lakesDiscreteFold1.arff and
tests on lakesDiscreteFold1.arff; my basic implementation gets 88.6% accuracy.  You should
expect to get about the same.  Once you get the basic algorithm working, modify it, so it uses
Laplace smoothing.

Note:

As we discussed in class, it is very convenient to have a jagged 3-dimensional array for
storing the conditional probabilities of the descriptive features, i.e. the primary index is
the target value, the secondary index is which descriptive feature, and the tertiary index is
what value that feature takes on.  The length of the innermost array is dependent on the
descriptive feature.

predicting: pH
"""

import re
from typing import Dict, List, Tuple, Set
import NB


class Data:
    def __init__(self):
        self.attributes: Dict[str, str] = {}  # name -> type
        self.featureData: Dict[str, List] = {}  # name -> list of values
        self.numericStats: Dict[str, Dict[str, float]] = {}  # name -> {min, max}
        self.discreteValues: Dict[str, Set] = {}  # name -> set of possible values

    #Adds an attribute to data storage
    def addAtributes(self, name: str, attributeType: str):
        self.attributes[name] = attributeType
        self.featureData[name] = []

        #If discrete attribute get the possible values
        if '{' in attributeType:
            values = attributeType.strip('{}').split(',')
            self.discreteValues[name] = {v.strip() for v in values}

    #Adds a row of data values
    def addDataToRow(self, values: List[str]):
        for(name, value) in zip(self.attributes.keys(), values):
            attributeType = self.attributes[name]

            if 'numeric' in attributeType:
                try:
                    value = float(value)
                except ValueError:
                    print(f"Warning: Could not convert {value} to float for attribute {name}")
                    value = None
            self.featureData[name].append(value)

    #A helper function to calculate stats
    def calcStats(self):
        for name, attributeType in self.attributes.items():
            if 'numeric' in attributeType:
                values = [x for x in self.featureData[name] if x is not None]
                if values:
                    self.numericStats[name] = {
                        'min' : min(values),
                        'max' : max(values)
                  }
            #For discrete attributes that weren't previously defined
            elif name not in self.discreteValues and '{' not in attributeType:
                self.discreteValues[name] = set(self.featureData[name])
    def getFeatureType(self, attributeName):
        if attributeName in self.attributes:
            if 'numeric' in self.attributes[attributeName]:
                return 'numeric'
            else:
                return 'discrete'
        return None

#Look through an .arff file and return arffData object
def arffFile(filename: str) -> Data:
    arffData = Data()

    #used to check if after @data for actual data info
    dataSection = False

    #opens up the .arff file and goes through each line
    with open(filename, 'r') as f:
        for line in f:

            #removes comments and removes tailing whitespaces
            line = line.split('%')[0].strip()

            #incase of empty lines keep going
            if not line:
                continue

            # Look for attributes and what they are
            if line.startswith('@attribute'):
                match = re.match(r'@attribute\s+\'?([^\']+)\'?\s+([^\s].*)', line)
                if match:
                    attributeName, attributeType = match.groups()
                    arffData.addAtributes(attributeName.strip(), attributeType.strip())

            # Looks for data section
            elif line.lower().startswith('@data'):
                dataSection = True
                continue

            #If we're in data section get the values, separating by ,
            elif dataSection:
                values = [v.strip() for v in line.split(',')]

                #So long as values is the same length on attributes save the data
                if len(values) == len(arffData.attributes):
                    arffData.addDataToRow(values)

    #Calculate the stats once all data is grabbed
    arffData.calcStats()
    return arffData

#Calculates the accuracy of the model on test data
def evaluateModel(nbModel: NB.NaiveBays, testData: Data) -> float:
    correct = 0
    total = len(testData.featureData[nbModel.targetAttribute])

    #print("\nFirst few predictions: ")

    #Create predictions for each instance in test data
    for i in range(total):
        # Create instance dictionary
        instance = {attr: testData.featureData[attr][i] for attr in testData.attributes}

        # Get prediction
        prediction = nbModel.predict(instance)
        actual = testData.featureData[nbModel.targetAttribute][i]

        #if i < 10:
            #print(f"Predicted: {prediction}, Actual: {actual}")

        if prediction == actual:
            correct += 1
    return (correct / total)*100

#the main section of the program
def main():
    try:
        # Get the training data
        trainFile = input("Please input training file name (making sure to add extension): ")
        trainData = arffFile(trainFile)


        #Get test data
        testFile = input("Please input testing file name (making sure to add extension): ")
        testData = arffFile(testFile)

        #Get the target attribute from available attributes
        print("\nAvailable attributes: ", list(trainData.attributes.keys()))
        targetAttribute = input("Enter the target attribute name: ")

        # ask if laplace smoothing should be used, default is yes, also lowers the input from user
        useLaplace = input("\nUse Laplace smoothing? (y/n, default: y): ").lower()
        useLaplace = useLaplace != 'n' or 'no'

        #Print the target values
        print("\nTarget values in training: ", set(trainData.featureData[targetAttribute]))
        print("Target values in testing: ", set(testData.featureData[targetAttribute]))

        #Create KNN Model
        print("\nCreating Naive Bayes model...")
        nb = NB.NaiveBays(trainData, targetAttribute, use_Laplace=useLaplace)
        #Evaluate on the test data
        print("\nEvaluating on test data...")
        accuracy = evaluateModel(nb, testData)
        print(f"\nAccuracy on test data: {accuracy:.1f}%")

        #Option to print out the model
        if input("\nWould you like to see the details of the learned model? (y/n): ").lower() == 'y' or 'yes':
            nb.printModel()

    except FileNotFoundError as e:
        print(f"Error: File not found - {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    main()
