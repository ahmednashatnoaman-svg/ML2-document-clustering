"""Document Clustering — core library.

Modules:
    data_loader    — load 20 Newsgroups and Wikipedia People datasets
    preprocessing  — clean → tokenize → lemmatize pipeline
    features       — TF-IDF vectorization + TruncatedSVD (LSA)
    clustering     — K-Means, Hierarchical, GMM algorithms
    evaluation     — silhouette, Davies-Bouldin, Calinski-Harabasz metrics
    visualization  — PCA, t-SNE, dendrogram, elbow, heatmap plots
"""
