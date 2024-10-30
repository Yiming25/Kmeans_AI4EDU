import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="K-means clustering on a dataset specified by the user.")
parser.add_argument('--dataset', type=str, required=True, help="Path to the CSV file containing the dataset.")
parser.add_argument('--n_clusters', type=int, default=2, help="Number of clusters for K-means. Default is 2.")
args = parser.parse_args()

# Check if the file exists
if not os.path.isfile(args.dataset):
    raise FileNotFoundError(f"The specified file '{args.dataset}' does not exist.")

# Load the dataset from the provided CSV file
data = pd.read_csv(args.dataset).values

# K-means clustering
kmeans = KMeans(n_clusters=args.n_clusters)
kmeans.fit(data)

# Print the labels assigned to each data point
print("Labels assigned to each data point:")
print(kmeans.labels_)

# Print the coordinates of the cluster centers
print("\nCoordinates of cluster centers:")
print(kmeans.cluster_centers_)

# Print the inertia (sum of squared distances to the closest cluster center)
print("\nInertia (sum of squared distances to the closest cluster center):")
print(kmeans.inertia_)

# Plotting the results
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='x')
plt.show()
