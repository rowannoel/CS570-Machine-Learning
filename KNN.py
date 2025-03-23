"""
Rowan Noel-Rickert
Implements the K-Nearest Neighbors Algorithm

Can handle both numeric and categorical features
"""
import math
from typing import Dict, List, Set, Any
from collections import Counter

class KNN:
    def __init__(self, data, targetAttribute, k=3):
        """
        Initializes the KNN model

        :param data: contains the training data
        :param targetAttribute: what target to predict
        :param k: number of neighbors to consider (default: 3)
        """
        self.data = data
        self.targetAttribute = targetAttribute
        self.k = k

        #Stores the feature types for speed
        self.featureTypes = {}
        for attr in data.attributes:
            #skips the target attribute
            if attr != targetAttribute:
                self.featureTypes[attr] = data.getFeatureType(attr)

        #Normalize numeric features for training data
        self.normalizedData = self._normalizeData()

        #Store the indices of the features to use (all except target)
        self.featureIndices = [attr for attr in data.attributes if attr != targetAttribute]

    def _normalizeData(self):
        """
        Normalizes numeric features to [0,1] range for equal weighting
        :return: a dictionary of normalized feature values
        """
        normalized = {}

        for attr in self.featureTypes:
            if self.featureTypes[attr] == 'numeric':
                #check to see if we have min/max numbers
                if attr in self.data.numericStats:
                    minValue = self.data.numericStats[attr]['min']
                    maxValue = self.data.numericStats[attr]['max']
                    rangeValue = maxValue - minValue

                    #Avoid dividing by zero
                    if rangeValue == 0:
                        normalized[attr] = [0.5 for _ in self.data.featureData[attr]]
                    else:
                        normalized[attr] = [(x - minValue) / rangeValue if x is not None else 0.5
                                            for x in self.data.featureData[attr]]
                #if no stats available use raw values
                else:
                    normalized[attr] = self.data.featureData[attr]

            #for categorical features don't normalize
            else:
                normalized[attr] = self.data.featureData[attr]

        return normalized

    def normalizeInstance(self, instance):
        """
        Normalizes a single instance using the training data normalization
        """
        normalizedInstance = {}

        for attr in self.featureTypes:
            if attr in instance:
                if self.featureTypes[attr] == 'numeric':
                    if attr in self.data.numericStats:
                        minValue = self.data.numericStats[attr]['min']
                        maxValue = self.data.numericStats[attr]['max']
                        rangeValue = maxValue - minValue

                        if rangeValue == 0:
                            normalizedInstance[attr] = 0.5
                        else:
                            value = instance[attr]
                            if value is None:
                                normalizedInstance[attr] = 0.5
                            #clip values outside the training range
                            else:
                                value = max(minValue, min(maxValue, value))
                                normalizedInstance[attr] = (value - minValue) / rangeValue
                    else:
                        normalizedInstance[attr] = instance[attr]
                else:
                    normalizedInstance[attr] = instance[attr]

        return normalizedInstance

    def calculateDistance(self, instance1, instance2):
        """
        calculate distance between two instances
        Numeric features: euclidean distance
        for categorical features: simple match (0 if same, 1 if different)
        """
        distance = 0.0

        for attr in self.featureIndices:
            if attr in instance1 and attr in instance2:
                #euclidean distance for numeric features
                if self.featureTypes[attr] == 'numeric':
                    val1 = instance1[attr]
                    val2 = instance2[attr]

                    if val1 is not None and val2 is not None:
                        distance += (val1 - val2)**2
                    #Missing values add the maxium possible distance
                    else:
                        distance += 1.0
                #Simple match for categorical features
                else:
                    if instance1[attr] != instance2[attr] or instance1[attr] is None or instance2[attr] is None:
                        distance += 1.0
        return math.sqrt(distance)

    def predict(self, instance):
        """
        Predicts the class if an instance using KNN
        :param instance: Dictionary with attribute name -> value mappings
        :return: predicted class value
        """
        #normalize the instance
        normalizedInstance = self.normalizeInstance(instance)

        #calculate the ditances to all training instances
        distances = []
        for i in range(len(self.data.featureData[self.targetAttribute])):
            #create a dictionary for the training instance
            trainInstance = {attr: self.normalizedData[attr][i] for attr in self.featureIndices}

            #calculate distance
            dist = self.calculateDistance(normalizedInstance, trainInstance)
            targetValue = self.data.featureData[self.targetAttribute][i]

            distances.append((dist, targetValue))

        #sort by distance and get the k nearest
        distances.sort(key=lambda x: x[0])
        nearestNeighbors = distances[:self.k]

        #get the most common value from the k nearest neighbors
        neighborClasses = [neighbor[1] for neighbor in nearestNeighbors]
        prediction = Counter(neighborClasses).most_common(1)[0][0]

        return prediction