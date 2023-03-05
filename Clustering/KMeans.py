import numpy as np
import matplotlib.pyplot as plt
import cv2

# Function to obtain the clusters of each pixel
def setClusters(data, centroids, k):
    clusters = np.zeros(data.shape[0], dtype = int)

    # Loop for calculating distances
    for i in range(len(data)):

        # Array to store the distances of a pixel_i against the centroids.
        distances = np.zeros(k)

        # Loop for calculating distances using Euclidean distance
        for j in range(k):
            distances[j] = np.sum((data[i] - centroids[j]) ** 2) ** 0.5
            
        # Store the cluster with the smallest distance corresponding to pixel_i
        clusters[i] = np.argmin(distances)
        
    return clusters


# Function to update the centroid values
def updateCentroids(data, centroids, clusters):
    # Keep the old centroid values
    old_centroids = centroids.copy()

    # Loop for calculating the average of the pixels related to a centroid according to its cluster.
    for i in range(len(centroids)):
        centroids[i] = np.mean(data[clusters == i, :], axis = 0)
    
    return old_centroids, centroids


# Function to compare old vs new centroids
def equal_centroids(old_centroids, new_centroids):
    for i in range(len(old_centroids)):
        if np.array_equal(old_centroids[i], new_centroids[i]) == False:
            return False
            
    return True


def kmeans(data, k):
    # Generate k centroids randomly from the data.
    centroids = []
    for i in range(k):
        centroids.append(data[np.random.randint(len(data))])

    centroids = np.array(centroids, dtype = int)

    # Loop for generating new clusters
    while True:
        # Obtain the clusters corresponding to each pixel
        clusters = setClusters(data, centroids, k)

        # Update the centroids
        old_centroids, centroids = updateCentroids(data, centroids, clusters)

        # When the centroids no longer change, the algorithm is over
        if (equal_centroids(old_centroids, centroids)): break
        
    # Relate the clusters to the centroids to segment the pixels into their corresponding group.
    return centroids[clusters]

    

def main():
    img = cv2.imread("images/doggie.jpeg") # Load image
    img = cv2.cvtColor(img, code = cv2.COLOR_BGR2RGB) # Convert RGB to BGR format
    data = np.reshape(img, (-1, img.shape[2])) # Reshape the image

    # Number of clusters (regions)
    k = 3

    segmented_data = kmeans(data, k)

    # Reshape the segmented data with the dimensions of the original image
    segmented_image = np.reshape(segmented_data, img.shape)

    txt = "K = " + str(k)
    plt.title(txt)
    plt.imshow(segmented_image)
    plt.show()


if __name__ == '__main__':        
    main()