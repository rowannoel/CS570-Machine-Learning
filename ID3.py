"""
Name: Rowan Noel-Rickert

Creates a decision tree using ID3 and uses pre-pruning to get better accuracy. Currently gets around
77%-80.4% accuracy. Range created after running a total of 10 times consecutively while running on the
"types" attribute
"""
import math
from typing import Dict, List, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
# Represents a node in the decision tree
class Node:
    # the splitting attribute at this node
    attribute: str = None
    # is this a leaf node or not
    isLeaf: bool = False
    # The classification for leaf nodes
    value: str = None
    # dictionary mapping attribute values to child nodes
    children: Dict[str, 'Node'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}


class ID3:
    # Initialize ID3 with data and target attributes
    def __init__(self, data, targetAttribute: str, maxDepth=10, minSamples=5):
        self.data = data
        self.targetAttribute = targetAttribute
        self.maxDepth = maxDepth
        self.minSamples = minSamples
        # makes sure no numerical attributes are there
        for attr, attrType in data.attributes.items():
            if 'numeric' in attrType:
                raise ValueError(f"Numerical attribute '{attr}' found. This only supports discrete attributes.")
        # get all attributes except the target
        self.attributes = set(data.attributes.keys()) - {targetAttribute}

    # Calculates the entropy for a set of data indiecs
    def entropy(self, dataIndices: List[int]) -> float:
        targetValues = [self.data.featureData[self.targetAttribute][i] for i in dataIndices]
        counts = Counter(targetValues)
        total = len(dataIndices)
        entropy = 0
        for count in counts.values():
            prob = count / total
            entropy -= prob * math.log2(prob)
        return entropy

    # Calcultes information gain for an attribute
    def informationGain(self, dataIndices: List[int], attribute: str) -> float:
        # Calculate entropy before split
        initialEntropy = self.entropy(dataIndices)
        # Group data indices by attribute values
        valueIndices = {}
        for indx in dataIndices:
            value = self.data.featureData[attribute][indx]
            valueIndices.setdefault(value, []).append(indx)
        # Calculates weighted entropy after split
        weightedEntropy = 0
        total = len(dataIndices)
        for indices in valueIndices.values():
            weight = len(indices) / total
            weightedEntropy += weight * self.entropy(indices)
        totalEntropy = initialEntropy - weightedEntropy
        return totalEntropy

    # Return the most common target value in the dataset
    def majorityValue(self, dataIndices: List[int]) -> str:
        targetValues = [self.data.featureData[self.targetAttribute][i] for i in dataIndices]
        return Counter(targetValues).most_common(1)[0][0]

    # Checks to make sure all examples have the same target values
    def allSameClass(self, dataIndices: List[int]) -> bool:
        return len(set(self.data.featureData[self.targetAttribute][i] for i in dataIndices)) == 1

    # Recursive algorithm to build decision tree
    def buildTree(self, dataIndices: List[int], availableAttributes: Set[str], depth=0) -> Node:
        # If all examples have same class, return leaf node
        # Added pruning
        # Check pruning conditions
        if (depth >= self.maxDepth or len(
                dataIndices) < self.minSamples or not availableAttributes or self.allSameClass(dataIndices)):
            return Node(isLeaf=True, value=self.majorityValue(dataIndices))
        # Find best attribute to split on
        bestGain = -1
        bestAttribute = None
        for attr in availableAttributes:
            gain = self.informationGain(dataIndices, attr)
            if gain > bestGain:
                bestGain = gain
                bestAttribute = attr
        # if no info gain make this a leaf node
        if bestGain <= 0:
            return Node(isLeaf=True, value=self.majorityValue(dataIndices))
        # Create a node for this split
        node = Node(attribute=bestAttribute)
        # Create child nodes for each value of the best attribute
        for value in self.data.discreteValues[bestAttribute]:
            childIndices = [i for i in dataIndices if self.data.featureData[bestAttribute][i] == value]
            # if no examples in this value create leaf with majority class
            if not childIndices:
                node.children[value] = Node(isLeaf=True, value=self.majorityValue(dataIndices))
            # recursively build subtree
            else:
                remainingAttributes = availableAttributes - {bestAttribute}
                node.children[value] = self.buildTree(childIndices, remainingAttributes, depth + 1)
        return node

    # train the decision tree on the full dataset
    def train(self) -> Node:
        dataIndices = list(range(len(self.data.featureData[self.targetAttribute])))
        return self.buildTree(dataIndices, self.attributes)

    # Predicts class for a single instance
    def predict(self, tree: Node, instance: Dict[str, str]) -> str:
        if tree.isLeaf:
            return tree.value
        # Gets the value of the spitting attribute for this instance
        value = instance[tree.attribute]
        # If we haven't seen this value from training return majority class
        if value not in tree.children:
            return self.majorityValue(list(range(len(self.data.featureData[self.targetAttribute]))))
        return self.predict(tree.children[value], instance)

    # Print the decision tree structure
    def printTree(self, node: Node, indent: str = "") -> None:
        if node.isLeaf:
            print(f"{indent}Predict: {node.value}")
            return
        print(f"{indent}Split on {node.attribute}")
        for value, child in node.children.items():
            print(f"{indent}If {node.attribute} = {value}:")
            self.printTree(child, indent + " ")