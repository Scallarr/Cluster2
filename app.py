# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:06:42 2025
@author: yoyop
"""

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Sidebar for user interaction
st.sidebar.header("🧪 Configure Clustering")
num_clusters = st.sidebar.slider("Select number of Clusters", min_value=2, max_value=10, value=3)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
y_kmeans = kmeans.fit_predict(X_pca)

# Define custom colors for clusters
cluster_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plotting the clusters
plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(
        X_pca[y_kmeans == i, 0], 
        X_pca[y_kmeans == i, 1], 
        c=cluster_colors[i], 
        s=50, 
        label=f"Cluster {i}"
    )



# Add titles and labels
plt.title("Clusters (2D PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()

# Show plot in Streamlit
st.pyplot(plt, use_container_width=True)
