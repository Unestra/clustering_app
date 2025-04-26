# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:55:40 2025

@author: LAB
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")
st.title("🔍 k-Means Clustering Visualizer by Araya Suchaichit")
st.subheader("📊 Example Data for Visualization")
st.markdown("This demo uses example data (2D) to illustrate clustering results.")

# Generate data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict
y_kmeans = loaded_model.predict(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot centroids
centers = loaded_model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.9, label='Centroids')

# Add legend and title
plt.title("k-Means Clustering")
plt.legend(loc='upper right')

# Show in Streamlit
st.pyplot(plt)

