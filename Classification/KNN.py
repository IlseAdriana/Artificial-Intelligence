import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import statistics as st
import matplotlib.pyplot as plt

def knn(tr_patterns, tr_labels, patterns, labels, k, n, testing = False):
    # Sets size
    tr_size = len(tr_patterns)
    pa_size = len(patterns)
    
    # Predictions vector of labels-length
    accuracy = np.zeros(np.size(np.unique(tr_labels)), dtype=int)

    # If testing is TRUE, it means that the test set is being received, 
    # so a sample of n patterns must be created.
    if (testing):
        samp_size = n # Sample size
        # Random indexes of patterns to be extracted from the test set
        te_rows = np.random.randint(0, pa_size, samp_size) 

        # A list to store the pattern index, its prediction and the real label
        results = np.zeros((samp_size, 3), dtype=int)
        curr_row = 0 # Counter to know the current row of the pattern
  

    # Loop for predicting each of the patterns
    for i in range(pa_size):
        # Array to store distances and associated labels
        distances = np.zeros((tr_size, 2))

        # Loop for calculating distances using Euclidean distance
        for j in range(tr_size):
            distances[j, 0] = sum((patterns[i,:] - tr_patterns[j,:]) ** 2) ** 0.5
            distances[j, 1] = tr_labels[j]
    
        # Sort distances in ascending order
        distances = distances[distances[:, 0].argsort()]
    
        # Obtain the first k distances
        k_neighbors = distances[:k, :]
    
        # Obtain the most repeated label
        predicted_label = int(st.mode(k_neighbors[:, 1]))
    
        # Check if the predicted label matches the real label
        if (predicted_label == labels[i]):
          # Add 1 to the predicted label value count
          accuracy[predicted_label] += 1

        # Check if the patterns match the testing set and 
        # also the current value o i is in the generated random indexes.
        if (testing and i in te_rows):
            # Add current pattern to the list of sample results
            results[curr_row] = i, predicted_label, labels[i]
            curr_row = curr_row + 1

    if (testing):
        print(f'Test set accuracy: {np.round(np.sum(accuracy)/pa_size, 2)}')

        # Display the sample results
        for i in range(samp_size):
            txt = "Predicted=" + str(results[i][1]) + " | " + "Real=" + str(results[i][2])
            plt.matshow(np.reshape(patterns[results[i][0]],(8,8)))
            plt.title(txt)
            plt.gray()
            plt.show()
    else:
        print(f'Training set accuracy: {np.round(np.sum(accuracy)/pa_size, 2)}')

def main():
    # Load patterns and labels from the dataset
    patterns, labels = load_digits(return_X_y = True)

    # Split the data into training and test set
    tr_patterns, te_patterns, tr_labels, te_labels = train_test_split(patterns, labels, test_size = 0.1, stratify = labels)

    # Number of Nearest Neighbors
    k = 3

    # Sample size
    n = 4

    knn(tr_patterns, tr_labels, tr_patterns, tr_labels, k, n) # Training set
    knn(tr_patterns, tr_labels, te_patterns, te_labels, k, n, True) # Testing set

    
if __name__ == "__main__":    
    main()