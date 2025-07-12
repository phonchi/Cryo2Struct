"""
Advanced clustering methods for Cryo2Struct atom predictions.

This module provides improved clustering algorithms that leverage prediction probabilities
and handle varying local densities better than simple distance-based clustering.

Created on Dec 2024
@author: AI Assistant
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')


class WeightedPoint:
    """Point with associated probability for clustering."""
    
    def __init__(self, x: float, y: float, z: float, prob: float):
        self.x = x
        self.y = y
        self.z = z
        self.prob = prob
    
    def to_array(self):
        """Convert to numpy array for clustering algorithms."""
        return np.array([self.x, self.y, self.z])


class ClusteringResult:
    """Container for clustering results and metrics."""
    
    def __init__(self, clusters: List[List[WeightedPoint]], centroids: List[Tuple[float, float, float, float]], 
                 method: str, params: Dict[str, Any]):
        self.clusters = clusters
        self.centroids = centroids
        self.method = method
        self.params = params
        self.n_clusters = len(clusters)
        self.metrics = {}
    
    def add_metric(self, name: str, value: float):
        """Add a quality metric to the result."""
        self.metrics[name] = value


class AdvancedClusteringMethods:
    """Advanced clustering methods for atom predictions."""
    
    def __init__(self):
        self.methods = {
            'dbscan': self._dbscan_clustering,
            'gmm': self._gmm_clustering,
            'weighted_kmeans': self._weighted_kmeans_clustering,
            'hierarchical': self._hierarchical_clustering,
            'adaptive_dbscan': self._adaptive_dbscan_clustering,
            'probability_weighted_dbscan': self._probability_weighted_dbscan,
            'auto': self.select_best_method,
        }
    
    def cluster_points(self, points: List[WeightedPoint], method: str = 'dbscan', 
                      **kwargs) -> ClusteringResult:
        """
        Cluster points using specified method.
        
        Args:
            points: List of WeightedPoint objects
            method: Clustering method name
            **kwargs: Method-specific parameters
            
        Returns:
            ClusteringResult object
        """
        if method == 'auto':
            return self.select_best_method(points, **kwargs)
        
        if method not in self.methods:
            raise ValueError(f"Unknown clustering method: {method}. Available: {list(self.methods.keys())}")
        
        if not points:
            return ClusteringResult([], [], method, kwargs)
        
        return self.methods[method](points, **kwargs)
    
    def _dbscan_clustering(self, points: List[WeightedPoint], eps: float = 2.0, 
                          min_samples: int = 2, use_probabilities: bool = True) -> ClusteringResult:
        """
        DBSCAN clustering with optional probability weighting.
        
        Args:
            points: List of WeightedPoint objects
            eps: Maximum distance between two samples for them to be considered as in the same neighborhood
            min_samples: Number of samples in a neighborhood for a point to be considered as a core point
            use_probabilities: Whether to use probabilities as sample weights
        """
        coords = np.array([p.to_array() for p in points])
        probs = np.array([p.prob for p in points])
        
        # Apply probability weighting by duplicating high-probability points
        if use_probabilities:
            weights = (probs / probs.min()).astype(int)
            weighted_coords = []
            weighted_points = []
            
            for i, (coord, point, weight) in enumerate(zip(coords, points, weights)):
                for _ in range(max(1, weight)):
                    weighted_coords.append(coord)
                    weighted_points.append(point)
            
            coords = np.array(weighted_coords)
            clustering_points = weighted_points
        else:
            clustering_points = points
        
        # Perform DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(coords)
        
        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise points
                continue
            if label not in clusters:
                clusters[label] = []
            
            # For weighted clustering, add original points not duplicates
            if use_probabilities:
                original_idx = i % len(points)
                if points[original_idx] not in clusters[label]:
                    clusters[label].append(points[original_idx])
            else:
                clusters[label].append(clustering_points[i])
        
        cluster_list = list(clusters.values())
        centroids = self._compute_centroids(cluster_list)
        
        result = ClusteringResult(cluster_list, centroids, 'dbscan', 
                                {'eps': eps, 'min_samples': min_samples, 'use_probabilities': use_probabilities})
        
        # Add quality metrics
        if len(cluster_list) > 1:
            try:
                silhouette = silhouette_score(coords, labels)
                result.add_metric('silhouette_score', silhouette)
            except:
                pass
            
            try:
                calinski = calinski_harabasz_score(coords, labels)
                result.add_metric('calinski_harabasz_score', calinski)
            except:
                pass
        
        return result
    
    def _gmm_clustering(self, points: List[WeightedPoint], n_components: int = None, 
                       covariance_type: str = 'full', random_state: int = 42) -> ClusteringResult:
        """
        Gaussian Mixture Model clustering.
        
        Args:
            points: List of WeightedPoint objects
            n_components: Number of components (estimated if None)
            covariance_type: Type of covariance parameters
            random_state: Random state for reproducibility
        """
        coords = np.array([p.to_array() for p in points])
        probs = np.array([p.prob for p in points])
        
        # Estimate number of components if not provided
        if n_components is None:
            n_components = max(1, min(len(points) // 3, int(np.sqrt(len(points)))))
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                             random_state=random_state)
        
        # Fit GMM without sample weights (not supported in all sklearn versions)
        gmm.fit(coords)
        labels = gmm.predict(coords)
        
        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
        
        cluster_list = list(clusters.values())
        centroids = self._compute_centroids(cluster_list)
        
        result = ClusteringResult(cluster_list, centroids, 'gmm', 
                                {'n_components': n_components, 'covariance_type': covariance_type})
        
        # Add quality metrics
        result.add_metric('bic', gmm.bic(coords))
        result.add_metric('aic', gmm.aic(coords))
        if len(cluster_list) > 1:
            try:
                silhouette = silhouette_score(coords, labels)
                result.add_metric('silhouette_score', silhouette)
            except:
                pass
        
        return result
    
    def _weighted_kmeans_clustering(self, points: List[WeightedPoint], n_clusters: int = None, 
                                   random_state: int = 42, max_iter: int = 300) -> ClusteringResult:
        """
        Weighted K-means clustering.
        
        Args:
            points: List of WeightedPoint objects
            n_clusters: Number of clusters (estimated if None)
            random_state: Random state for reproducibility
            max_iter: Maximum number of iterations
        """
        coords = np.array([p.to_array() for p in points])
        probs = np.array([p.prob for p in points])
        
        # Estimate number of clusters if not provided
        if n_clusters is None:
            n_clusters = max(1, min(len(points) // 5, int(np.sqrt(len(points)))))
        
        # Weighted K-means using sample weights
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
        
        # Apply probability weighting
        weighted_coords = []
        weighted_points = []
        weights = (probs / probs.min() * 10).astype(int)
        
        for i, (coord, point, weight) in enumerate(zip(coords, points, weights)):
            for _ in range(max(1, weight)):
                weighted_coords.append(coord)
                weighted_points.append(point)
        
        weighted_coords = np.array(weighted_coords)
        labels = kmeans.fit_predict(weighted_coords)
        
        # Group original points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            
            original_idx = i % len(points)
            if points[original_idx] not in clusters[label]:
                clusters[label].append(points[original_idx])
        
        cluster_list = list(clusters.values())
        centroids = self._compute_centroids(cluster_list)
        
        result = ClusteringResult(cluster_list, centroids, 'weighted_kmeans', 
                                {'n_clusters': n_clusters, 'random_state': random_state})
        
        # Add quality metrics
        result.add_metric('inertia', kmeans.inertia_)
        if len(cluster_list) > 1:
            try:
                silhouette = silhouette_score(weighted_coords, labels)
                result.add_metric('silhouette_score', silhouette)
            except:
                pass
        
        return result
    
    def _hierarchical_clustering(self, points: List[WeightedPoint], n_clusters: int = None, 
                               linkage: str = 'ward', distance_threshold: float = None) -> ClusteringResult:
        """
        Hierarchical clustering.
        
        Args:
            points: List of WeightedPoint objects
            n_clusters: Number of clusters (estimated if None)
            linkage: Linkage criterion
            distance_threshold: Distance threshold for clustering
        """
        coords = np.array([p.to_array() for p in points])
        
        # Estimate number of clusters if not provided
        if n_clusters is None and distance_threshold is None:
            n_clusters = max(1, min(len(points) // 4, int(np.sqrt(len(points)))))
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage, 
            distance_threshold=distance_threshold
        )
        labels = hierarchical.fit_predict(coords)
        
        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
        
        cluster_list = list(clusters.values())
        centroids = self._compute_centroids(cluster_list)
        
        result = ClusteringResult(cluster_list, centroids, 'hierarchical', 
                                {'n_clusters': n_clusters, 'linkage': linkage, 'distance_threshold': distance_threshold})
        
        # Add quality metrics
        if len(cluster_list) > 1:
            try:
                silhouette = silhouette_score(coords, labels)
                result.add_metric('silhouette_score', silhouette)
            except:
                pass
        
        return result
    
    def _adaptive_dbscan_clustering(self, points: List[WeightedPoint], k: int = 3, 
                                  percentile: float = 90) -> ClusteringResult:
        """
        Adaptive DBSCAN that estimates eps based on k-distance graph.
        
        Args:
            points: List of WeightedPoint objects
            k: Number of neighbors to consider for eps estimation
            percentile: Percentile of k-distances to use as eps
        """
        coords = np.array([p.to_array() for p in points])
        
        # Handle edge cases
        if len(points) <= k:
            # Fall back to regular DBSCAN with a reasonable eps
            return self._dbscan_clustering(points, eps=2.0, min_samples=max(1, len(points)//2), use_probabilities=True)
        
        # Build k-distance graph
        tree = cKDTree(coords)
        k_distances = []
        
        for coord in coords:
            distances, _ = tree.query(coord, k=k+1)  # +1 because query includes the point itself
            k_distances.append(distances[k])  # k-th nearest neighbor distance
        
        # Estimate eps as percentile of k-distances
        k_distances = np.array(k_distances)
        eps = np.percentile(k_distances, percentile)
        
        # Ensure eps is valid
        if np.isnan(eps) or eps <= 0:
            eps = np.mean(k_distances) if not np.isnan(np.mean(k_distances)) else 2.0
        
        min_samples = max(2, k)
        
        # Perform DBSCAN with estimated parameters
        return self._dbscan_clustering(points, eps=eps, min_samples=min_samples, use_probabilities=True)
    
    def _probability_weighted_dbscan(self, points: List[WeightedPoint], base_eps: float = 2.0, 
                                   prob_weight: float = 0.5, min_samples: int = 2) -> ClusteringResult:
        """
        DBSCAN with probability-weighted distance metric.
        
        Args:
            points: List of WeightedPoint objects
            base_eps: Base epsilon value
            prob_weight: Weight for probability in distance calculation
            min_samples: Minimum samples for core point
        """
        coords = np.array([p.to_array() for p in points])
        probs = np.array([p.prob for p in points])
        
        # Create probability-weighted distance matrix
        n_points = len(points)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    spatial_dist = np.linalg.norm(coords[i] - coords[j])
                    prob_factor = 1.0 / (1.0 + prob_weight * (probs[i] + probs[j]) / 2.0)
                    distances[i, j] = spatial_dist * prob_factor
        
        # Custom DBSCAN using precomputed distances
        labels = np.full(n_points, -1)
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] != -1:  # Already processed
                continue
            
            # Find neighbors
            neighbors = np.where(distances[i] <= base_eps)[0]
            
            if len(neighbors) < min_samples:
                labels[i] = -1  # Noise
                continue
            
            # Start new cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)
            
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                
                if labels[q] == -1:  # Noise point
                    labels[q] = cluster_id
                elif labels[q] != -1:  # Already in another cluster
                    j += 1
                    continue
                
                labels[q] = cluster_id
                
                # Find neighbors of q
                q_neighbors = np.where(distances[q] <= base_eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend(q_neighbors)
                
                j += 1
            
            cluster_id += 1
        
        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
        
        cluster_list = list(clusters.values())
        centroids = self._compute_centroids(cluster_list)
        
        result = ClusteringResult(cluster_list, centroids, 'probability_weighted_dbscan', 
                                {'base_eps': base_eps, 'prob_weight': prob_weight, 'min_samples': min_samples})
        
        return result
    
    def _compute_centroids(self, clusters: List[List[WeightedPoint]]) -> List[Tuple[float, float, float, float]]:
        """
        Compute probability-weighted centroids for clusters.
        
        Args:
            clusters: List of clusters, each containing WeightedPoint objects
            
        Returns:
            List of (x, y, z, avg_prob) tuples
        """
        centroids = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            # Compute weighted centroid
            total_weight = sum(p.prob for p in cluster)
            if total_weight == 0:
                total_weight = len(cluster)
            
            weighted_x = sum(p.x * p.prob for p in cluster) / total_weight
            weighted_y = sum(p.y * p.prob for p in cluster) / total_weight
            weighted_z = sum(p.z * p.prob for p in cluster) / total_weight
            avg_prob = sum(p.prob for p in cluster) / len(cluster)
            
            centroids.append((weighted_x, weighted_y, weighted_z, avg_prob))
        
        return centroids
    
    def select_best_method(self, points: List[WeightedPoint], methods: List[str] = None, 
                          metric: str = 'silhouette_score') -> ClusteringResult:
        """
        Automatically select the best clustering method based on quality metrics.
        
        Args:
            points: List of WeightedPoint objects
            methods: List of methods to try (default: all available)
            metric: Metric to optimize ('silhouette_score', 'calinski_harabasz_score', etc.)
            
        Returns:
            Best ClusteringResult based on the specified metric
        """
        if methods is None:
            methods = ['dbscan', 'adaptive_dbscan', 'probability_weighted_dbscan', 'gmm', 'weighted_kmeans']
        
        best_result = None
        best_score = float('-inf')
        
        for method in methods:
            try:
                result = self.cluster_points(points, method)
                if metric in result.metrics:
                    score = result.metrics[metric]
                    if score > best_score:
                        best_score = score
                        best_result = result
            except Exception as e:
                print(f"Warning: Method {method} failed: {e}")
                continue
        
        return best_result if best_result else self.cluster_points(points, 'dbscan')
    
    def get_clustering_report(self, result: ClusteringResult) -> str:
        """
        Generate a report for clustering results.
        
        Args:
            result: ClusteringResult object
            
        Returns:
            String report of clustering results
        """
        report = []
        report.append(f"Clustering Method: {result.method}")
        report.append(f"Parameters: {result.params}")
        report.append(f"Number of Clusters: {result.n_clusters}")
        report.append(f"Total Points Clustered: {sum(len(cluster) for cluster in result.clusters)}")
        
        if result.metrics:
            report.append("Quality Metrics:")
            for metric, value in result.metrics.items():
                report.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(report)