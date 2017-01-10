from brain import Brain
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
