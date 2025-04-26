# # -*- coding: utf-8 -*-
# """
# Created on Sun Apr 20 15:55:40 2025

# @author: LAB
# """

# import streamlit as st
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# # Load model
# with open('kmeans_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

# # Page config
# st.set_page_config(page_title="k-Means Clustering App", layout="centered")
# st.title("🔍 k-Means Clustering Visualizer by Araya Suchaichit")
# st.subheader("📊 Example Data for Visualization")
# st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# # Generate data
# X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# # Predict
# y_kmeans = loaded_model.predict(X)

# # Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# # Plot centroids
# centers = loaded_model.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.9, label='Centroids')

# # Add legend and title
# plt.title("k-Means Clustering")
# plt.legend(loc='upper right')

# # Show in Streamlit
# st.pyplot(plt)
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:14:00 2025

@author: Nongnuch
"""

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# Page title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Sidebar - Number of clusters
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 3)

# Run K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
reduced_df["Cluster"] = labels

# Plot clusters
fig, ax = plt.subplots()
for cluster in range(k):
    cluster_data = reduced_df[reduced_df["Cluster"] == cluster]
    ax.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()

# Show plot and data
st.pyplot(fig)
st.dataframe(reduced_df.head(10))
