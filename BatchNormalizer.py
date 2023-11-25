import numpy as np

class FitError(Exception):
    pass

class BatchNormalizer:
    def __init__(self, gamma=1, beta=0):
        self.gamma = gamma
        self.beta = beta
    
    def fit(self, samples):
        self.mean = [samples[:, i].mean() for i in range(samples.shape[0])]
        self.std = [samples[:, i].std() for i in range(samples.shape[0])]

    def _check_is_fitted(self):
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise FitError("Must call fit before transforming data!")
  
    def transform(self, x):
        self._check_is_fitted()
        return self.gamma * (x - self.mean) / self.std + self.beta

    def inverse_transform(self, x):
        self._check_is_fitted()
        return (x - self.beta) * self.std / self.gamma + self.mean
