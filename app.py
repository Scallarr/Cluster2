# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:06:42 2025

@author: yoyop
"""

# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Title
st.title("🔍 k-Means Clustering Visualizer")

# Display section header
st.subheader("📊 Interactive Cluster Selection")
st.markdown("Use the slider below to choose the number of clusters for synthetic data.")

# Slider to select number of clusters
num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=4)

# Generate synthetic data
X, _ = make_blobs(
    n_samples=300,
    centers=num_clusters,
    cluster_std=0.60,
    random_state=0
)

# Fit a new KMeans model on generated data
kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
kmeans_model.fit(X)

# Predict cluster labels
y_kmeans = kmeans_model.predict(X)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.title(f'K-Means Clustering with {num_clusters} Clusters')

# Display cluster centers
centers = kmeans_model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='o', label="Centroids")

plt.legend()
st.pyplot(plt)
