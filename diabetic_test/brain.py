import os
import sys
from sys import argv
import pickle

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain import TanhLayer, SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import FeedForwardNetwork, FullConnection, LinearLayer, RecurrentNetwork

# Local Imports
from conf import *

class Brain():
    """docstring for Brain."""
    def __init__(self):
        self.net = RecurrentNetwork()

        inLayer = LinearLayer(8)
        hiddenLayer = SigmoidLayer(30)
        outLayer = LinearLayer(2)

        self.net.addInputModule(inLayer)
        self.net.addModule(hiddenLayer)
        self.net.addOutputModule(outLayer)

        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)

        self.net.addConnection(in_to_hidden)
        self.net.addConnection(hidden_to_out)

        self.net.sortModules()

        self.alldata = ClassificationDataSet(8, 1, nb_classes = 2)
        self.traindata = ClassificationDataSet(8, 1, nb_classes = 2)
        self.testdata = ClassificationDataSet(8, 1, nb_classes = 2)
        self.c_trainer = BackpropTrainer(self.net, self.alldata, verbose=True)

    def addTrainData(self):
        self.addSamples(read_csv(path=DBPATH))
        self.addSamples(read_csv(path=NDBPATH))

    def addSamples(self, rows):
        for row in rows:
            attributes = list(map(float, row[:8]))
            if (row[8] == "1"):
                self.addSample(attributes, [1])
            else:
                self.addSample(attributes, [0])

    def addSample(self, sample, group):
        self.alldata.addSample(sample, group)

    def accuracy(self):
        if len(self.alldata) == 0:
            print "No data_sets found. Maybe you loaded the classifier from a file?"
            return

        tstresult = percentError(
                        self.c_trainer.testOnClassData(dataset=self.alldata),
                        self.alldata['class']
                    )

        print "epoch: %4d" % self.c_trainer.totalepochs, \
              "trainer error: %5.2f%%" % tstresult, \
              "trainer accuracy: %5.2f%%" % (100-tstresult)

    def trainMachine(self):
        self.alldata._convertToOneOfMany()
        self.testdata, self.traindata = self.alldata.splitWithProportion(SPLIT_PROPORTION)
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
        if result == 1:
            print "+ve"
        else :
            print "-ve"

        return result
