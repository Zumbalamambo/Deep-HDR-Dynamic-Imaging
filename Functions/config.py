class Params:
    def __init__(self):
        self.testNet = 'TrainedModel'
        self.gamma = 2.2
        self.mu = 5000
        self.patchSize = 40
        self.stride = 20
        self.cropSizeTraining = 50
        self.batchSize = 20
        self.numAugment = 10
        self.numTotalAugment = 48
        self.border = 6
        self.weps = 1e-6
        self.trainingScenes = 'TrainingData/Training/'
        self.trainingData = 'TrainingData/Training/'
        self.testScenes = 'TrainingData/Test/'
        self.testData = 'TrainingData/Test/'
        self.alpha = 0.0001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8


def set_params():
    m = Params()
    return m
