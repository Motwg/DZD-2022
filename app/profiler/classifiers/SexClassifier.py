from torch import nn
from app.profiler.classifiers.Classifier import Classifier


class SexClassifier(Classifier):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.layers = nn.Sequential(
            nn.Linear(97, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.layers(x)
