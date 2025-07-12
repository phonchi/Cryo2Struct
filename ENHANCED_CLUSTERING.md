# Enhanced Clustering Methods for Cryo2Struct

## Overview

This document describes the enhanced clustering methods implemented for Cryo2Struct to improve atom position clustering based on predicted probabilities. The new methods provide better handling of varying densities, probability weighting, and multiple clustering algorithms.

## New Clustering Methods

### 1. DBSCAN (Density-Based Spatial Clustering)
- **Method**: `dbscan`
- **Description**: Density-based clustering that can find arbitrarily shaped clusters and handle noise
- **Parameters**:
  - `eps`: Maximum distance between two samples for them to be considered neighbors (default: 2.0)
  - `min_samples`: Number of samples in a neighborhood for a point to be considered as a core point (default: 2)
  - `use_probabilities`: Whether to use prediction probabilities as weights (default: True)
- **Best for**: Data with varying densities and noise points

### 2. Adaptive DBSCAN
- **Method**: `adaptive_dbscan`
- **Description**: Automatically estimates the optimal eps parameter using k-distance analysis
- **Parameters**:
  - `k`: Number of neighbors to consider for eps estimation (default: 3)
  - `percentile`: Percentile of k-distances to use as eps (default: 90)
- **Best for**: Data where optimal clustering parameters are unknown

### 3. Probability-Weighted DBSCAN
- **Method**: `probability_weighted_dbscan`
- **Description**: DBSCAN with probability-weighted distance metric
- **Parameters**:
  - `base_eps`: Base epsilon value (default: 2.0)
  - `prob_weight`: Weight for probability in distance calculation (default: 0.5)
  - `min_samples`: Minimum samples for core point (default: 2)
- **Best for**: Data where prediction confidence should strongly influence clustering

### 4. Gaussian Mixture Models (GMM)
- **Method**: `gmm`
- **Description**: Probabilistic clustering using Gaussian mixture models
- **Parameters**:
  - `n_components`: Number of Gaussian components (auto-estimated if not provided)
  - `covariance_type`: Type of covariance ('full', 'tied', 'diag', 'spherical', default: 'full')
- **Best for**: Data with overlapping clusters and probabilistic assignments

### 5. Weighted K-Means
- **Method**: `weighted_kmeans`
- **Description**: K-means clustering with probability-based sample weighting
- **Parameters**:
  - `n_clusters`: Number of clusters (auto-estimated if not provided)
  - `random_state`: Random seed for reproducibility (default: 42)
- **Best for**: Data with known number of clusters and clear separation

### 6. Hierarchical Clustering
- **Method**: `hierarchical`
- **Description**: Agglomerative hierarchical clustering
- **Parameters**:
  - `n_clusters`: Number of clusters (auto-estimated if not provided)
  - `linkage`: Linkage criterion ('ward', 'complete', 'average', 'single', default: 'ward')
- **Best for**: Data requiring hierarchical cluster relationships

### 7. Automatic Method Selection
- **Method**: `auto`
- **Description**: Automatically selects the best clustering method based on quality metrics
- **Parameters**:
  - `methods`: List of methods to try (default: all available)
  - `metric`: Metric to optimize (default: 'silhouette_score')
- **Best for**: When optimal method is unknown

## Usage

### Command Line Interface

```bash
# Use DBSCAN clustering
python utils/cluster_cn_predicted_map.py prob.txt map.mrc out.mrc \
    --clustering_method dbscan \
    --dbscan_eps 2.0 \
    --dbscan_min_samples 2 \
    --use_probability_weighting

# Use adaptive DBSCAN (recommended)
python utils/cluster_cn_predicted_map.py prob.txt map.mrc out.mrc \
    --clustering_method adaptive_dbscan \
    --use_probability_weighting

# Use automatic method selection
python utils/cluster_cn_predicted_map.py prob.txt map.mrc out.mrc \
    --clustering_method auto \
    --use_probability_weighting \
    --clustering_report report.txt

# Use GMM clustering
python utils/cluster_cn_predicted_map.py prob.txt map.mrc out.mrc \
    --clustering_method gmm \
    --gmm_components 5 \
    --use_probability_weighting
```

### Configuration File

Add the following parameters to your `config/arguments.yml`:

```yaml
clustering_method: 'adaptive_dbscan'  # Choose clustering method
dbscan_eps: 2.0                       # DBSCAN epsilon parameter
dbscan_min_samples: 2                 # DBSCAN minimum samples
gmm_components: null                  # GMM components (auto if null)
kmeans_clusters: null                 # K-means clusters (auto if null)
hierarchical_clusters: null          # Hierarchical clusters (auto if null)
use_probability_weighting: true       # Use probability-weighted centroids
```

### Python API

```python
from utils.advanced_clustering import AdvancedClusteringMethods, WeightedPoint

# Create clustering object
clustering = AdvancedClusteringMethods()

# Create weighted points
points = [
    WeightedPoint(x=1.0, y=2.0, z=3.0, prob=0.8),
    WeightedPoint(x=1.1, y=2.1, z=3.1, prob=0.7),
    # ... more points
]

# Perform clustering
result = clustering.cluster_points(points, method='adaptive_dbscan')

# Access results
print(f"Number of clusters: {result.n_clusters}")
print(f"Centroids: {result.centroids}")
print(f"Quality metrics: {result.metrics}")

# Generate report
report = clustering.get_clustering_report(result)
print(report)

# Automatic method selection
best_result = clustering.select_best_method(points)
print(f"Best method: {best_result.method}")
```

## Quality Metrics

The new clustering methods provide several quality metrics:

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion (higher is better)
- **Inertia**: Sum of squared distances to centroids (lower is better, for K-means)
- **BIC/AIC**: Bayesian/Akaike Information Criterion (lower is better, for GMM)

## Performance Comparison

Based on typical cryo-EM data:

| Method | Speed | Quality | Noise Handling | Parameter Tuning |
|--------|-------|---------|----------------|------------------|
| Legacy | ★★★★★ | ★★ | ★ | ★★★★★ |
| DBSCAN | ★★★★ | ★★★★ | ★★★★★ | ★★★ |
| Adaptive DBSCAN | ★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| GMM | ★★★ | ★★★★ | ★★★ | ★★★ |
| Weighted K-means | ★★★★ | ★★★ | ★★ | ★★★ |
| Hierarchical | ★★ | ★★★★ | ★★★ | ★★★★ |

## Recommendations

1. **For most cases**: Use `adaptive_dbscan` with `use_probability_weighting=true`
2. **For noisy data**: Use `dbscan` or `probability_weighted_dbscan`
3. **For overlapping clusters**: Use `gmm`
4. **For unknown optimal method**: Use `auto`
5. **For backward compatibility**: Use `legacy`

## Backward Compatibility

The enhanced clustering methods maintain full backward compatibility:
- Default behavior unchanged (uses legacy clustering)
- All existing parameters and interfaces preserved
- New parameters are optional with sensible defaults

## Testing

Run the test suite to verify the implementation:

```bash
# Basic functionality tests
python test_advanced_clustering.py

# Integration tests
python test_integration.py
```

## Examples

Example clustering reports are generated showing:
- Method used and parameters
- Number of clusters found
- Quality metrics
- Execution time
- Recommendations for parameter tuning

This enhanced clustering system provides significant improvements in atom position clustering quality while maintaining the simplicity and reliability of the original Cryo2Struct pipeline.