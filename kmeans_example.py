   import numpy as np​

   from sklearn.cluster import KMeans​

   import matplotlib.pyplot as plt​

   ​

   # Example dataset​

   data = np.array([[1, 2], [2, 3], [3, 4], [5, 8], [8, 8], [9, 11]])​

   ​

   # K-means clustering​

   kmeans = KMeans(n_clusters=2)​

   kmeans.fit(data)​

  # Print the labels assigned to each data point ​

  print("Labels assigned to each data point:") ​

  print(kmeans.labels_) ​

  # Print the coordinates of the cluster centers ​

  print("\nCoordinates of cluster centers:") ​

  print(kmeans.cluster_centers_) ​

  # Print the inertia (sum of squared distances to the closest cluster center) ​

  print("\nInertia (sum of squared distances to the closest cluster center):") ​

  print(kmeans.inertia_)​

   ​

   # Plotting the results​

   plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='rainbow')​

   plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='x')​

   plt.show()
