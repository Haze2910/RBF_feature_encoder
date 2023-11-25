import numpy as np

class RBFFeatureEncoder:
    def __init__(self, centers, sigma=0.5):
        self.centers = centers
        self.sigma = sigma

    def encode(self, x):
        """
        Calculate the norm between the inputs and the centers and return the RBF encoded features for each input i.e 
        
            out_i = exp( -(||x - c_i|| ** 2) / 2 * sigma ** 2)
            
        Input: (n, input_size)
        Output: (n, centers.shape[0])
        """
        norms = np.linalg.norm((x[:, np.newaxis, :] - self.centers), axis=2) ** 2 
        return (np.exp((-norms / (2 * self.sigma ** 2))))

    @property
    def size(self):
        """
        number of output features i.e number of centers
        """
        return self.centers.shape[0]
