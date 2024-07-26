# Aircraft Regime Detection

The objective of this project is to accurately classify different phases of an aircraft's flight based on data obtained from the Automatic Dependent Surveillance-Broadcast (ADS-B) system. ADS-B data includes information such as altitude, speed, rate of climb, and other parameters that describe the aircraft's state during its flight.
Identifying and categorizing these phases, such as climb, cruise, descent, and approach, is crucial for monitoring the aircraft's performance, safety, and maintenance needs. This process, known as aircraft regime detection, helps in predicting potential failures, optimizing maintenance schedules, and ensuring the overall safety of flight operations.

Solution Approach
To address this problem, unsupervised learning techniques were employed to analyze and cluster the flight data into different phases. The following clustering algorithms were used:

K-Means Clustering:

K-Means is a partition-based clustering method that divides the data into a predefined number of clusters, specified by the parameter K.
It aims to minimize the within-cluster variance by iteratively updating the cluster centroids and reassigning data points to the nearest centroid.
In this project, K-Means was applied with different values of K (4, 5, 6, 7, 8) to determine the optimal number of clusters for phase detection.

Birch (Balanced Iterative Reducing and Clustering using Hierarchies):

Birch is a hierarchical clustering method that builds a clustering feature tree (CF tree) for the data incrementally.
It is particularly useful for large datasets and can handle noise effectively.
Birch clusters the data hierarchically and can be adjusted to produce the desired number of clusters, making it suitable for this problem.

Gaussian Mixture Models (GMM):

GMM is a probabilistic clustering algorithm that assumes the data is generated from a mixture of several Gaussian distributions.
It provides a soft assignment of data points to clusters based on the posterior probabilities.
GMM was used to capture the underlying distribution of the flight data and classify it into different phases based on the estimated parameters of the Gaussian distributions.
These clustering algorithms were applied to the encoded flight data, transformed using an autoencoder, to identify distinct flight phases and analyze their performance in terms of phase detection accuracy.
