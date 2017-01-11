from brain import Brain
from conf import *

class DiabeticTest():
    def __init__(self):
        self.brain = Brain()

    def save(self):
        self.brain.save()

    def load(self):
        self.brain.load()

    def train(self):
        self.brain.addTrainData()
        self.brain.trainMachine()

    def test(self):
        print("Please Enter the patient details for test : ")
        attributes = []
        attributes.append(input("Number of times pregnant : "))
        attributes.append(input("Plasma glucose concentration : "))
        attributes.append(input("Blood pressure : "))
        attributes.append(input("Skin thikness : "))
        attributes.append(input("Insulin unit : "))
        attributes.append(input("BMI : "))
        attributes.append(input("Diabeties predigree function : "))
        attributes.append(input("Age : "))
        self.brain.classify(attributes)

    def classify(self):
        total = 0
        right = 0
        wrong = 0

        for row in read_csv():
            total += 1
            attributes = list(map(float, row[:8]))

            if (row[8] == "1"):
                result = self.brain.classify(attributes)
                if result == 1:
                    right += 1
                else:
                    wrong += 1
                print "ACTUAL: Diabetic"
            else:
                result = self.brain.classify(attributes)
                if result == 0:
                    right += 1
                else:
                    wrong += 1
                print "ACTUAL: Not Diabetic"

        print "total: %4d" % total, \
              "right: %4d" % right, \
              "wrong: %4d" % wrong
