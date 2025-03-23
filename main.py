"""
Name: Rowan Noel-Rickert

Write an implementation of ID3 using your dataset reader from the first homework.
It only needs to handle discrete-valued features (for both target and descriptive features).
You can either ignore numerical features or exit with an error if a numerical feature is detected in the dataset.
Provide a main which trains it on lakesDiscreteFold1.arff, tests on lakesDiscreteFold2.arff, and outputs the accuracy
(these datasets are available in the notes).  For comparison, my no-frills-implementation gets an accuracy of 84.2%
so you should expect to get about the same.
"""

import re
from typing import Dict, List, Tuple, Set
import ID3


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
def evaluateModel(tree: ID3.Node, testData: Data, id3Model: ID3.ID3) -> float:
    correct = 0
    total = len(testData.featureData[id3Model.targetAttribute])

    #print("\nFirst few predictions: ")

    #Create predictions for each instance in test data
    for i in range(total):
        # Create instance dictionary
        instance = {attr: testData.featureData[attr][i] for attr in testData.attributes}

        # Get prediction
        prediction = id3Model.predict(tree, instance)
        actual = testData.featureData[id3Model.targetAttribute][i]

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
        #print("\nAvailable attributes: ", list(trainData.attributes.keys()))
        targetAttribute = input("Enter the target attribute name: ")

        print("Target values in training:", set(trainData.featureData[targetAttribute]))
        print("Target values in training:", set(testData.featureData[targetAttribute]))

        #Create and train ID3 Model
        print("\nTraining ID3 decision tree...")
        id3 = ID3.ID3(trainData, targetAttribute, maxDepth=5)
        tree = id3.train()

        #Print the tree structure
        print("\nDecision Tree Structure: ")
        id3.printTree(tree)

        #Evaluate on the test data
        print("\nEvaluating on test data...")
        accuracy = evaluateModel(tree, testData, id3)
        #formats it so it only goes to 1 decimal place and is a floating number
        print(f"\nAccuracy on test data: {accuracy:.1f}%")

        return trainData, tree
    except FileNotFoundError as e:
        print(f"Error: File not found - {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    main()
