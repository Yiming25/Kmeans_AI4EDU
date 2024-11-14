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

# Extract the dataset name without extension for file naming
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

# K-means clustering
kmeans = KMeans(n_clusters=args.n_clusters)
kmeans.fit(data)

# Save the clustering results to a .txt file
with open(f"{dataset_name}_results.txt", "w") as file:
    file.write("Labels assigned to each data point:\n")
    file.write(f"{kmeans.labels_}\n\n")
    
    file.write("Coordinates of cluster centers:\n")
    file.write(f"{kmeans.cluster_centers_}\n\n")
    
    file.write("Inertia (sum of squared distances to the closest cluster center):\n")
    file.write(f"{kmeans.inertia_}\n")

print(f"Clustering results saved to {dataset_name}_results.txt")

# Plotting the results
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='x')
plt.title(f"K-means Clustering ({dataset_name})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save the figure as a .png file with the dataset name
plt.savefig(f"{dataset_name}_clusters.png")
plt.show()

print(f"Clustering plot saved to {dataset_name}_clusters.png")
