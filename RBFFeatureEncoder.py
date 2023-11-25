import numpy as np

class RBFFeatureEncoder:
    def __init__(self, centers, sigma=0.5):
        self.centers = centers
        self.sigma = sigma

    def encode(self, state): # modify
        # Calculate the norm between the state and the centers and return the RBF encoded features
        norms = np.linalg.norm(state - centers), axis=1) ** 2 
        return (np.exp( (-norms/(2 * self.sigma ** 2)) )).flatten()

    @property
    def size(self): 
        return self.centers.shape[0]
