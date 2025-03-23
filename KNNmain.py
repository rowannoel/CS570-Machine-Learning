"""
Name: Rowan Noel-Rickert

Write an implementation of KNN, able to handle both categorical and continuous descriptive features, and categorical
target features.  Provide a main which trains it on lakesFold1.arff, tests on lakesFold2.arff (these datasets are
available in the notes), and outputs the accuracy with "K" set to 5.  Here are some results that my
no-frills-implementation gets for comparison:

    Train on lakesFold1.arff, test on lakesFold2.arff, k=5:  64.6% accuracy
    Train on lakesFold1.arff, test on lakesFold2.arff, k=1:  64.4% accuracy
    Train and test on lakesFold1.arff, k=1:  100% accuracy
    Train and test on lakesFold1.arff, k=5:  78.6% accuracy

You should expect to get about the same.  Once your implementation is done, make it straightforward to get it to run
with another value of K (e.g. by setting a parameter or changing a constant).  It isn't necessary to do distance
weighting; you can simply take the mode of the K nearest instances.  It also isn't necessary to scale the features as
part of your implementation (though you might try having Weka do this as a pre-processing step and see how that affects
your accuracy).

Hints:

To get an implementation working for K=1, you are just iterating through all the training instances, calculating
distances to the instance you are trying to predict, and using the target value of whatever training instance was
closest.  Categorical descriptive features can be dealt with in a number of ways, but here is one simple method you can
use: if the two discrete values for a given feature are identical, treat that as distance 0 in that dimension,
otherwise treat it as distance 1 in that dimension.

To get it to work for K > 1, I recommend modifying the K=1 code in the following manner.  As you iterate through all
the training instances calculating distances to some new instance you are trying to predict, instead of storing the
closest target value observed so far, store (distance, targetValue) pairs in an array (or some data structure that
maintains sorted order).  Once the array is populated, you can sort it by distance (increasing) and look at the first K
target values to determine the most common target value.  That's it.  Just be careful not to overrun your data structure
if K exceeds the number of training instances.

Note that it is relatively straightforward to adapt your classifier to predict numeric values.  Instead of taking the
mode of the K nearest target values, you would report their mean (not required).
"""

import re
from typing import Dict, List, Tuple, Set
import KNN


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
def evaluateModel(knnModel: KNN.KNN, testData: Data) -> float:
    correct = 0
    total = len(testData.featureData[knnModel.targetAttribute])

    #print("\nFirst few predictions: ")

    #Create predictions for each instance in test data
    for i in range(total):
        # Create instance dictionary
        instance = {attr: testData.featureData[attr][i] for attr in testData.attributes}

        # Get prediction
        prediction = knnModel.predict(instance)
        actual = testData.featureData[knnModel.targetAttribute][i]

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

        kValue = int(input("Enter the k value for KNN: "))

        #Print the target values
        print("\nTarget values in training: ", set(trainData.featureData[targetAttribute]))
        print("Target values in testing: ", set(testData.featureData[targetAttribute]))

        #Create KNN Model
        print("\nCreating KNN model...")
        knn = KNN.KNN(trainData, targetAttribute, k = kValue)

        #Evaluate on the test data
        print("\nEvaluating on test data...")
        accuracy = evaluateModel(knn, testData)
        print(f"\nAccuracy on test data: {accuracy:.1f}%")

    except FileNotFoundError as e:
        print(f"Error: File not found - {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    main()
