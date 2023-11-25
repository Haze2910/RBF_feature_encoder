import numpy as np
import BatchNormalizer, RBFFeatureEncoder

def main():
    x_dim = 5 # dimension of the data
  
    n_samples = 10 # number of samples to calculate mean and std of
    samples = np.random.random((n_samples, x_dim))

    # Initialize the normalizer and fit the samples
    normalizer = BatchNormalizer()
    normalizer.fit(samples)

    n_data = 20 # number of data points
    x = np.random.random((n_data, x_dim))

    # Normalize the data
    normalized_x = normalizer.transform(x) 

    n_centers = 10 # number of centers i.e. number of output feature for each data point
    centers = np.random.random((n_centers, x_dim))

    # Initialize the encoder
    encoder = RBFFeatureEncoder(centers)

    # Encode each data point
    encoded_x = encoder.encode(normalized_x)
    print(f"Encoded data:\n {encoded_x}")



if __name__ == "__main__":
    main()
    
    
