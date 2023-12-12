# -*- coding: utf-8 -*-
"""M22EE051_Task1.ipynb

## Perform K-means clustering on MNIST data from scratch. Instead of using Euclidian distance as  distance metric use Cosine Similarity as distance metric. Clustering should be done in 10, 7, and 4 clusters.Visualize the images getting clustered in different clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeansCosine:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Initialize centroids using KMeans++
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            # Assign clusters using cosine similarity
            cluster_group = self.assign_clusters(X)

            old_centroids = self.centroids

            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # Check for convergence
            if np.all(old_centroids == self.centroids):
                break

        return cluster_group

    def initialize_centroids(self, X):
        # Using KMeans++ initialization
        centroids = [X[np.random.choice(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            similarities = np.array([np.dot(x, c) / (np.linalg.norm(x) * np.linalg.norm(c)) for c in centroids for x in X])
            similarities = similarities.reshape(len(centroids), -1)
            probabilities = similarities.sum(axis=0) / similarities.sum()
            centroids.append(X[np.random.choice(len(X), p=probabilities)])

        return np.array(centroids)

    def assign_clusters(self, X):
        # Cosine similarity calculation
        similarities = np.array([[np.dot(x, c) / (np.linalg.norm(x) * np.linalg.norm(c)) for c in self.centroids] for x in X])
        cluster_group = np.argmax(similarities, axis=1)
        return cluster_group

    def move_centroids(self, X, cluster_group):
        # Calculate new centroids using the mean of cosine similarity
        new_centroids = np.array([X[cluster_group == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

# Load the CSV file
data = pd.read_csv('/content/mnist_train.csv')

# Extract true labels from the first column
true_labels = data.iloc[:, 0]

# Drop the first column (labels) before clustering
image_data = data.drop(data.columns[0], axis=1).values

image_data

# Normalize the pixel values to the range [0, 1]
image_data_normalized = image_data / 255.0
image_data_normalized

# Visualize the images in different clusters
def visualize_clusters(kmeans, clusters, num_clusters):
    plt.figure(figsize=(12, 8))
    for i in range(min(10, num_clusters)):
        cluster_indices = np.where(clusters == i)[0]
        for j in range(min(5, len(cluster_indices))):
            plt.subplot(5, num_clusters, j * num_clusters + i + 1)
            reshaped_image = image_data[cluster_indices[j]].reshape(28, 28) #reshaping image to original size 28*28 for visualization
            plt.imshow(reshaped_image, cmap='gray')
            plt.title(f'Cluster {i}')
            plt.axis('off')
    plt.show()

kmeans_10_clusters = KMeansCosine(n_clusters=10, max_iter=100)
clusters_10 = kmeans_10_clusters.fit_predict(image_data_normalized)
visualize_clusters(kmeans_10_clusters, clusters_10, 10)

kmeans_7_clusters = KMeansCosine(n_clusters=7, max_iter=100)
clusters_7 = kmeans_7_clusters.fit_predict(image_data_normalized)
visualize_clusters(kmeans_7_clusters, clusters_7, 7)

kmeans_4_clusters = KMeansCosine(n_clusters=4, max_iter=100)
clusters_4 = kmeans_4_clusters.fit_predict(image_data_normalized)
visualize_clusters(kmeans_4_clusters, clusters_4, 4)

"""## Try to write a python function which finds optimal number of clusters for this dataset?"""

def find_optimal_clusters(data, max_clusters=20):
    distortions = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeansCosine(n_clusters=i)
        labels = kmeans.fit_predict(data)
        centroids = np.array([data[labels == j].mean(axis=0) for j in range(i)])
        distortions.append(np.sum(np.linalg.norm(data - centroids[labels], axis=1)))

    # Plot the elbow graph
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

# Use this function with your normalized image data
find_optimal_clusters(image_data_normalized)
