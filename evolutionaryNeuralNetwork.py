##################################################
# A script to implement an evolutionary neural network
# Uses the sigmoid function to calculate the
# activation of each node
#
# Note: this is mostly the same as my other neural
# network code, just with a reproduction function
#
# Date: 9.1.2018
#
# Author: Patrick Naughton
##################################################

import sys
import random
import math


def main():
    ev = Evolution(10, [4, 5, 3], 1, 1, 0.5)
    for i in range(100):
        ev.doGeneration()
        ev.assessAllFitness()
        print(ev.fitnessScores)

class Evolution:
    def __init__(self, numCreatures, creatureArchitecture, weightRange, volatilityRange, percentSurvival):
        """
        Initialize the evolution simulator
        :param numCreatures: number of creatures per generation
        :param creatureArchitechture: defines the architecture of each network (same as layernums)
        :param weightRange: range of potential starting weights
        :param volatilityRange: range of potential volatilities
        """
        self.numCreatures = numCreatures
        self.creatureArchitecture = creatureArchitecture
        self.weightRange = weightRange
        self.volatilityRange = volatilityRange
        if percentSurvival < 1 and percentSurvival > 0:
            self.percentSurvival = percentSurvival
        else:
            self.percentSurvival = 0.5
        self.creatures = []
        self.fitnessScores = []
        self.generation = 0
        for i in range(numCreatures):
            volatility = self.volatilityRange - 2*self.volatilityRange*random.random()
            self.creatures.append(NeuralNetwork(self.creatureArchitecture, self.weightRange, volatility))

    def doGeneration(self):
        self.generation += 1
        self.assessAllFitness()
        self.sortFitness()
        survivors = self.creatures[:int(len(self.creatures)*self.percentSurvival)]
        ind = 0
        while len(survivors) < self.numCreatures:
            if ind < len(survivors):
                survivors.append(survivors[ind].reproduce())
            else:
                ind = 0
        self.creatures = survivors

    def massExtinction(self, percentageSurvival):
        """
        Randomly kill 1-percentageSurvival creatures
        to mimic a mass extinction
        :param percentageSurvival: the percent of creatures who survive
        :return: None, just update self.creatures
        """
        survivors = []
        if percentageSurvival<1 and percentageSurvival>0:
            for i in range(int(self.numCreatures*percentageSurvival)):
                ind = random.choice(range(len(self.creatures)))
                survivors.append(self.creatures.pop(ind))
        ind = 0
        while len(survivors) < self.numCreatures:
            if ind < len(survivors):
                newBorn = survivors[ind].reproduce()
                survivors.append(newBorn)
                ind += 1
            else:
                ind = 0
        self.creatures = survivors

    def assessAllFitness(self):
        """
        Assess the fitness of all creatures
        :return: None, just update self.fitnessScores
        """
        self.fitnessScores = []
        for i in range(len(self.creatures)):
            self.fitnessScores.append(self.assessFitness(i))

    def assessFitness(self, creatureID):
        """
        Assess the fitness of the creature at index creatureID
        :param creatureID: the index of the creature
        :return: a double representing the creature's fitness
        """
        input = [[random.random()]]
        for i in range(1, self.creatureArchitecture[0]):
            input.append([2 * input[i-1][0]])
        ans = self.creatures[creatureID].generateOutput(input)
        return (ans[0][0] - ans[1][0])**2

    def sortFitness(self):
        """
        Sort creatures (in decreasing order) based on fitness
        :return: none, simply update self.creatures and self.fitnessScores
        """
        indicies = range(len(self.fitnessScores))
        indicies = self.sort(indicies)
        creatureAns = []
        fitnessAns = []
        for i in indicies:
            creatureAns.append(self.creatures[i])
            fitnessAns.append(self.fitnessScores[i])
        self.creatures = creatureAns
        self.fitnessScores = fitnessAns

    def sort(self, arr):
        """
        Use helper function merge to return arr sorted based on fitness scores
        We use merge sort because it's expected to have to compute many
        computations, so we want a fast algorithm
        :param arr: array of indicies to sort
        :return: arr, sorted based on self.fitnessScores
        """
        if len(arr) <= 1:
            return(arr)
        else:
            return(self.merge(self.sort(arr[:len(arr)//2]), self.sort(arr[len(arr)//2:])))

    def merge(self, arr1, arr2):
        """
        merge two sorted lists such that the final
        list is sorted (in decreasing order)
        :param arr1: array of indicies of creatures/fitness levels
        :param arr2: array of indicies of creatures/fitness levels
        :return: a sorted merged list
        """
        ans = []
        ind1 = 0
        ind2 = 0
        while ind1<len(arr1) and ind2<len(arr2):
            if self.fitnessScores[arr1[ind1]] > self.fitnessScores[arr2[ind2]]:
                ans.append(arr1[ind1])
                ind1 += 1
            else:
                ans.append(arr2[ind2])
                ind2 += 1
        while ind1<len(arr1):
            ans.append(arr1[ind1])
            ind1 += 1
        while ind2<len(arr2):
            ans.append(arr2[ind2])
            ind2 += 1
        return ans


class NeuralNetwork:
    def __init__(self, layerNums, weightRange, volatility):
        """
        Initialize the neural network
        ::params::
        layerNums is a list or tuple with
        the number of nodes in each layer
        (the first entry being the input layer
        the last the output layer)"""
        self.volatility = volatility
        self.layerNums = layerNums
        self.weightRange = weightRange
        self.layers = []
        self.numInputs = layerNums[0]
        self.numOutputs = layerNums[len(layerNums) - 1]
        for i in range(len(layerNums)):
            layer = Layer(layerNums[i])
            if not i == len(layerNums) - 1:
                layer.generateRandomWeights(layerNums[i + 1], weightRange)
            self.layers.append(layer)

    #TODO define a volatileReproduce that can generate new nodes or entire layers

    def reproduce(self):
        # Randomly change the next creature's volatility
        newNN = NeuralNetwork(self.layerNums, self.weightRange,
                              self.volatility*1+(self.volatility-2*self.volatility*random.random()))
        for L in range(len(self.layers) - 1):
            for i in range(len(self.layers[L].weights)):
                for j in range(len(self.layers[L].weights[i])):
                    # Randomly multiply each weight
                    newNN.layers[L].weights[i][j] = (self.layers[L].weights[i][j] *
                                                     (1 + (self.volatility - 2 * self.volatility * random.random())))
        return newNN

    def setWeights(self, weights):
        if len(weights) == len(self.layers) - 1:
            for i in range(len(self.layers) - 1):
                if len(weights[i]) == self.layers[i + 1].numNodes and len(weights[i][0]) == self.layers[i].numNodes + 1:
                    self.layers[i].weights = weights[i]
                else:
                    print("Error in setWeights: weight dimension is wrong for weight at index " + str(i))
            self.layers[len(self.layers) - 1].weights = [[]]
        else:
            print("Error in setWeights:\nWeight size did not match number of layers\nNumber of layers: "
                  + len(self.layers) + "\nWeight size: " + len(weights));

    def generateOutput(self, x):
        """
        After we've trained the network, return the output
        of a given input in column vector notation
        ::params::
        x is a column vector representing the input"""
        self.forwardPropagate(x)
        return (self.layers[len(self.layers) - 1].nodes)

    def train(self, x, y, learningRate, epsilon, maxIterations):
        """
        Train this neural network
        using inputs x mapped to outputs y
        ::params::
        x = a list of column vectors representing inputs
        y = a list of column vectors representing outputs"""
        w = open("weights.log", "w")
        for iter in range(maxIterations):
            w.write("Iteration " + str(iter) + " :\n")
            for L in range(len(self.layers)):
                w.write(str(self.layers[L].weights) + "\n\n")
            w.write("\n\n")
            print("Training is %10.10f%% done" % (100 * iter / maxIterations), end="\r")
            for L in range(len(self.layers)):
                self.layers[L].clearDelta()
            for i in range(len(y)):
                self.forwardPropagate(x[i])
                self.backPropagate(y[i])
            for L in range(len(self.layers)):
                for i in range(len(self.layers[L].weights)):
                    for j in range(len(self.layers[L].weights[i])):
                        self.layers[L].weights[i][j] -= learningRate * self.layers[L].delta[i][j] / len(y)
        print()
        for L in range(len(self.layers)):
            w.write(str(self.layers[L].weights) + "\n\n")
        w.close()

    def forwardPropagate(self, input):
        """
        Forward propagate through the neural network
        to produce an answer"""
        self.layers[0].nodes = input[:]
        x = input[:]
        for i in range(len(self.layers) - 1):
            x = self.layers[i].generateAnswers(x)
            self.layers[i + 1].nodes = x[:]
        return (x)

    def backPropagate(self, answer):
        """
        Back propagate through the network
        to calculate the gradient of the cost
        function wrt each weight. Update each layer's
        delta accordingly"""
        d = []
        outputError = []
        outputLayer = self.layers[len(self.layers) - 1].nodes
        for i in range(len(outputLayer)):
            outputError.append([outputLayer[i][0] - answer[i][0]])
        d.insert(0, outputError)
        for i in range(len(self.layers) - 2, 0, -1):
            newD = self.layers[i].matrixMultiply(self.layers[i].transpose(self.layers[i].weights), d[0])[1:]
            for j in range(len(newD)):
                newD[j][0] *= \
                self.layers[i].act_deriv(self.layers[i - 1].generateAnswers(self.layers[i - 1].nodes[:]))[j][0]
            d.insert(0, newD)
        for L in range(len(self.layers) - 1):
            for i in range(len(self.layers[L].nodes)):
                for j in range(len(d[L])):
                    self.layers[L].delta[j][i + 1] += self.layers[L].nodes[i][0] * d[L][j][0]
            for j in range(len(d[L])):
                self.layers[L].delta[j][0] += d[L][j][0]


class Layer:
    def __init__(self, numNodes, weights=[[]]):
        """
        Initialize the layer with numNodes nodes
        and the weights specified in weights where
        weights is a matrix with rows = numNodes
        of the next layer and columns = numNodes+1"""
        self.numNodes = numNodes
        self.weights = weights
        self.nodes = []
        self.delta = []
        self.clearDelta()

    def generateRandomWeights(self, nextLayerNumNodes, epsilon):
        """
        set self.weights = a matrix of random weights of
        the appropriate size
        Also update delta to reflect the size of weights"""
        self.weights = []
        for i in range(nextLayerNumNodes):
            row = []
            for j in range(self.numNodes + 1):
                row.append(epsilon - (2 * epsilon * random.random()))
            self.weights.append(row)
        self.clearDelta()

    def generateAnswers(self, input):
        """
        Generate the inputs to the next layer
        by (matrix) multiplying this layer's weights
        by input, where input is a column vector.
        Also, prepend input with a 1 (the constant term)
        ::params::
        input should be a column vector with the same number of
        rows as this layer has nodes"""
        input.insert(0, [1])
        ans = self.matrixMultiply(self.weights, input)
        for i in range(len(ans)):
            ans[i] = self.act(ans[i])
        return (ans)

    def clearDelta(self):
        self.delta = []
        for i in range(len(self.weights)):
            row = []
            for j in range(len(self.weights[i])):
                row.append(0)
            self.delta.append(row)

    def transpose(self, mat):
        """
        return the transpose of matrix mat"""
        if len(mat) == 0 or len(mat[0]) == 0:
            return ([[]])
        newMat = []
        for i in range(len(mat[0])):
            row = []
            for j in range(len(mat)):
                row.append(mat[j][i])
            newMat.append(row)
        return (newMat)

    def matrixMultiply(self, mat1, mat2):
        """
        Return the product of mat1 and
        mat2 if the dimensions match up"""
        ans = []
        if len(mat1) > 0 and len(mat2) > 0 and len(mat2[0]) > 0:
            if len(mat1[0]) == len(mat2):
                for i in range(len(mat1)):
                    row = []
                    for j in range(len(mat2[0])):
                        sum = 0
                        for k in range(len(mat1[i])):
                            sum += mat1[i][k] * mat2[k][j]
                        row.append(sum)
                    ans.append(row)
                return (ans)
            else:
                print("Colums not equal to rows")
                print(mat1)
                print(mat2)
                return (None)
        else:
            print("One or both matricies are empty")
            print(mat1)
            print(mat2)
            return (None)

    def act(self, z):
        """
        Return the activation of z,
        in this case, using the sigmoid
        function to determine activation"""
        return (self.sigmoid(z))

    def act_deriv(self, z):
        """
        Return the derivative of the activation
        function at the point z"""
        return (self.sigmoid_deriv(z))

    def sigmoid(self, z):
        """
        return the sigmoid of z"""
        try:
            if z >= 500:
                return (1)
            elif z <= -500:
                return (0)
            return (1 / (1 + math.exp(-z)))
        except:
            try:
                for i in range(len(z)):
                    if z[i] >= 500:
                        z[i] = 1
                    elif z[i] <= -500:
                        z[i] = 0
                    else:
                        z[i] = 1 / (1 + math.exp(-z[i]))
                return (z)
            except:

                for i in range(len(z)):
                    for j in range(len(z[i])):
                        if z[i][j] >= 500:
                            z[i][j] = 1
                        elif z[i][j] <= -500:
                            z[i][j] = 0
                        else:
                            z[i][j] = 1 / (1 + math.exp(-z[i][j]))
                return (z)

    def sigmoid_deriv(self, z):
        """
        return the derivative of the
        sigmoid at z"""
        try:
            if math.fabs(z) >= 500:
                return (0)
            return (math.exp(-z) / (1 + math.exp(-z)) ** 2)
        except:
            try:
                for i in range(len(z)):
                    if math.fabs(z) >= 500:
                        z[i] = 0
                    else:
                        z[i] = (math.exp(-z[i]) / (1 + math.exp(-z[i])) ** 2)
                return (z)
            except:
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        if math.fabs(z[i][j]) > 500:
                            z[i][j] = 0
                        else:
                            z[i][j] = (math.exp(-z[i][j]) / (1 + math.exp(-z[i][j])) ** 2)
                return (z)

    def disp(self, mode):
        """
        display this layer
        ::params::
        mode
            if mode = 0: print all the weights
            elif mode = 1: print all the nodes"""
        if mode == 0:
            print(self)
        elif mode == 1:
            for n in self.nodes:
                print(str(n) + "\n")

    def __str__(self):
        """
        print out all the weights of this layer
        in matrix format"""
        r = ""
        for w in self.weights:
            r += str(w) + "\n"
        return (r)


if __name__ == "__main__":
    main()