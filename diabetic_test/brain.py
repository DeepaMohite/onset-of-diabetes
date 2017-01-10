import csv
import os
import sys
from sys import argv
import pickle

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain import TanhLayer
from pybrain.structure.modules import SoftmaxLayer


class Brain():
    """docstring for Brain."""
    def __init__(self):
        self.net = buildNetwork(8, 30, 2, hiddenclass = TanhLayer)
        self.alldata = ClassificationDataSet(8, 1, nb_classes = 2)
        self.traindata = ClassificationDataSet(8, 1, nb_classes = 2)
        self.testdata = ClassificationDataSet(8, 1, nb_classes = 2)
        self.c_trainer = BackpropTrainer(self.net, self.alldata, verbose=True)

    def addTrainData(self):
         with open("/home/sushil/AI/SampleDBData.csv", "rb") as csvfile:
             spamreader = csv.reader(csvfile, delimiter=',')
             for row in spamreader:
                self.alldata.addSample(row[:8], row[8])

    def trainMachine(self):
        self.alldata._convertToOneOfMany()
        self.testdata, self.traindata = self.alldata.splitWithProportion(0.25)
        # self.c_trainer.trainUntilConvergence()
        self.c_trainer.trainEpochs(30)

    def save(self):
        with open("/home/sushil/AI/classifier.brain", "wb") as fp:
            pickle.dump(self.net, fp)

    def load(self):
        with open("/home/sushil/AI/classifier.brain", "rb") as fp:
            self.net = pickle.load(fp)

    def classify(self, testdata):
        score = self.net.activate(testdata)
        print(score)
        result = max(range(len(score)), key=score.__getitem__)
        if result == 1 :
             print "+ve"
        else :
             print "-ve"
         
