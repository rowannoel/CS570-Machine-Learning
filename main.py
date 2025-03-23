"""
Name: Rowan Noel-Rickert
Write a program that can read any .arff file which has only discrete and continuous features.
You choose the language.  It should:

    [x]Ignore comments, which start with the % character
    [x]Be able to read lakes.arff, provided here
    [x]Store the data internally (you will need it later for implementing ML algorithms)
    [x]Report the min and max for each numeric-valued feature
    [x]Report all possible values for each discrete-valued feature
    [X]Finally, find some other dataset that is of interest to you (or create one).
        Convert it to .arff format and make sure your program runs on it.  Include it here.

"""

import re
from typing import Dict, List, Tuple, Set

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

#the main section of the program
def main():
    try:
        filename = input("Please input file name (making sure to add extension): ")

        #Go through the arff file
        arffData = arffFile(filename)

        #Prints the attributes and their stats
        print("\nAttributes Stats ")
        for name, attributeType in arffData.attributes.items():
            print(f"\nAttribute: {name} ({attributeType})")

            if name in arffData.numericStats:
                stats = arffData.numericStats[name]
                print(f" Numeric Feature:")
                print(f" - Min: {stats['min']}")
                print(f" - Max: {stats['max']}")
            elif name in arffData.discreteValues:
                values = arffData.discreteValues[name]
                print(f" Discrete Feature:")
                print(f" - Possible values: {sorted(values)}")

        return arffData

    except FileNotFoundError:
      print(f"Error: File '{filename}' not found")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()