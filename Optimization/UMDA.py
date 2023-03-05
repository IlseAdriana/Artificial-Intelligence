import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to apply otsu method
def otsu_evaluation(frequencies, umbrales, k):
    # Check that the pixels values are between 0 and 255
    if (umbrales < 0).any() or (umbrales > 254).any():
        return 1000

    # Copy the thresholds and sort them
    umbrales = umbrales.copy()[np.argsort(umbrales)]

    # Check that there're no repeated thresholds.
    if(k != np.unique(umbrales).shape[0]):
        return 1000

    # Frequency sum
    prob_total = np.sum(frequencies[1])

    # Probability vector
    probabilities = frequencies[1] / prob_total

    w_k = np.zeros(k + 1) # Array to store the cumulative probability
    mu_k = np.zeros(k + 1) # Array to store the cumulative mean
    umb_val = 0  # Current threshold value

    # Loop for calculating cumulative probability and mean
    for i, umbral in enumerate(umbrales):
        w_k[i] = np.sum(probabilities[umb_val:umbral])
        mu_k[i] = np.sum(
            (frequencies[0][umb_val:umbral] * probabilities[umb_val:umbral]) / w_k[i]
        )
        umb_val = umbral

    # Calculate the values after the last threshold up to the upper limit
    w_k[k] = np.sum(probabilities[umb_val:255])
    mu_k[k] = np.sum(
        frequencies[0][umb_val:255] * (probabilities[umb_val:255]) / w_k[k]
    )

    # Calculate average intensity
    mu_t = np.sum(w_k * mu_k)

    # Calculate the variance between classes
    stdev_b = np.sum(w_k * np.power((mu_k - mu_t), 2))

    # Minimizing the inverse of the variance between classes
    return np.power(stdev_b, -1)


def umda(frequencies, n, m, gen, k):
    # Generate population
    population = np.random.randint(1, 254, (n, k))

    # Array to store the fitness of individuals
    fitness = np.zeros(n)

    # Evaluate population using otsu method
    for _ in range(gen):
        # Evaluate individuals as they were thresholds
        for i, individual in enumerate(population):
            fitness[i] = otsu_evaluation(frequencies, individual, k)

        # Obtain the first m-values indexes
        m_idx = np.argsort(fitness)[:m]

        # Obtain the mean and standard deviation of the first m individuals.
        mean_m = np.mean(population[m_idx], axis=0)
        stdev_m = np.std(population[m_idx], axis=0)

        # Generate new population from mean and standard deviation
        population = np.random.normal(mean_m, stdev_m, (n, k)).astype(int)

    return np.sort(population[np.argsort(fitness)[0]])


def main():
    img = cv2.imread("images/lexa.png") # Load image
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY) # Convert BGR to Gray scale

    k = 70 # Number of thresholds
    n = 90  # Population size
    m = 20  # Truncation value
    gen = 20  # Generations to be calculated

    # Obtain frequencies for histogram
    frequencies = np.unique(img, return_counts=True)

    # Calculate thresholds
    umbrales = umda(img, n, m, gen, k)
    print(f"Selected threshold: {umbrales}")

    # Display histogram
    plt.plot(frequencies[0], frequencies[1])
    for umbral in umbrales:
        plt.plot([umbrales, umbrales], [0, np.max(frequencies[1])])
    
    plt.show()

    linear_img = img.reshape(-1)  # Reestructuramos a un arreglo unidimensional

    # Image coloring
    linear_img[linear_img < umbrales[0]] = 0
    linear_img[linear_img >= umbrales[-1]] = 255
    if k != 1:
        for i, umbral in enumerate(umbrales[: k - 1]):
            linear_img[
                (linear_img >= umbral) == (linear_img < umbrales[i + 1])
            ] = umbral

    # Reshape to original dimension
    img = np.reshape(linear_img, img.shape)
    txt = "Number of thresholds = " + str(k)
    plt.title(txt)
    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
   main()