#!/usr/bin/env python3
"""
Test script for advanced clustering methods in Cryo2Struct.

This script tests the new clustering algorithms and compares their performance
with the original clustering method.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.advanced_clustering import AdvancedClusteringMethods, WeightedPoint
from utils.clustering_centroid import create_clusters, Point
import os
import tempfile
import time


def generate_test_data(n_clusters=5, n_points_per_cluster=20, noise_level=0.1):
    """Generate synthetic test data for clustering validation."""
    np.random.seed(42)
    
    # Generate cluster centers
    centers = []
    for i in range(n_clusters):
        center = np.random.uniform(-10, 10, 3)
        centers.append(center)
    
    # Generate points around each center
    points = []
    true_labels = []
    
    for cluster_id, center in enumerate(centers):
        for _ in range(n_points_per_cluster):
            # Add some noise around the center
            offset = np.random.normal(0, noise_level * 2, 3)
            point = center + offset
            
            # Random probability (higher for points closer to center)
            dist_to_center = np.linalg.norm(offset)
            prob = max(0.1, 1.0 - dist_to_center / (noise_level * 4))
            
            points.append(WeightedPoint(point[0], point[1], point[2], prob))
            true_labels.append(cluster_id)
    
    # Add some noise points
    n_noise = n_points_per_cluster // 4
    for _ in range(n_noise):
        point = np.random.uniform(-12, 12, 3)
        prob = np.random.uniform(0.1, 0.3)
        points.append(WeightedPoint(point[0], point[1], point[2], prob))
        true_labels.append(-1)  # Noise label
    
    return points, true_labels


def test_clustering_methods():
    """Test all clustering methods and compare results."""
    print("Testing Advanced Clustering Methods for Cryo2Struct")
    print("=" * 60)
    
    # Generate test data
    points, true_labels = generate_test_data(n_clusters=4, n_points_per_cluster=15)
    print(f"Generated {len(points)} test points with {len(set(true_labels))} true clusters")
    
    # Initialize clustering methods
    clustering = AdvancedClusteringMethods()
    
    # Test all methods
    methods_to_test = [
        'dbscan',
        'adaptive_dbscan',
        'probability_weighted_dbscan',
        'gmm',
        'weighted_kmeans',
        'hierarchical'
    ]
    
    results = {}
    
    for method in methods_to_test:
        print(f"\nTesting {method}...")
        start_time = time.time()
        
        try:
            result = clustering.cluster_points(points, method)
            elapsed_time = time.time() - start_time
            
            results[method] = {
                'result': result,
                'n_clusters': result.n_clusters,
                'n_points_clustered': sum(len(cluster) for cluster in result.clusters),
                'execution_time': elapsed_time,
                'metrics': result.metrics
            }
            
            print(f"  Method: {method}")
            print(f"  Clusters found: {result.n_clusters}")
            print(f"  Points clustered: {sum(len(cluster) for cluster in result.clusters)}")
            print(f"  Execution time: {elapsed_time:.4f} seconds")
            
            if result.metrics:
                print(f"  Quality metrics:")
                for metric_name, value in result.metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method] = {'error': str(e)}
    
    # Test legacy method for comparison
    print(f"\nTesting legacy clustering method...")
    start_time = time.time()
    
    # Convert to legacy Point objects
    legacy_points = [Point(p.x, p.y, p.z) for p in points]
    legacy_clusters = create_clusters(legacy_points, thres=2.0)
    legacy_time = time.time() - start_time
    
    results['legacy'] = {
        'n_clusters': len(legacy_clusters),
        'n_points_clustered': sum(len(cluster) for cluster in legacy_clusters),
        'execution_time': legacy_time,
        'metrics': {}
    }
    
    print(f"  Method: legacy")
    print(f"  Clusters found: {len(legacy_clusters)}")
    print(f"  Points clustered: {sum(len(cluster) for cluster in legacy_clusters)}")
    print(f"  Execution time: {legacy_time:.4f} seconds")
    
    # Test auto method selection
    print(f"\nTesting automatic method selection...")
    start_time = time.time()
    best_result = clustering.select_best_method(points)
    auto_time = time.time() - start_time
    
    print(f"  Best method selected: {best_result.method}")
    print(f"  Clusters found: {best_result.n_clusters}")
    print(f"  Execution time: {auto_time:.4f} seconds")
    if best_result.metrics:
        print(f"  Quality metrics:")
        for metric_name, value in best_result.metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print(f"{'Method':<25} {'Clusters':<10} {'Points':<10} {'Time(s)':<10} {'Silhouette':<12}")
    print("-" * 70)
    
    for method, result in results.items():
        if 'error' in result:
            print(f"{method:<25} {'ERROR':<10} {'':<10} {'':<10} {'':<12}")
        else:
            silhouette = result['metrics'].get('silhouette_score', 'N/A')
            if silhouette != 'N/A':
                silhouette = f"{silhouette:.4f}"
            
            print(f"{method:<25} {result['n_clusters']:<10} {result['n_points_clustered']:<10} "
                  f"{result['execution_time']:<10.4f} {silhouette:<12}")
    
    return results


def test_real_data_format():
    """Test clustering with realistic atom prediction data format."""
    print("\nTesting with realistic atom prediction data format...")
    
    # Simulate data format from probabilities_atom.txt
    # Format: [[x, y, z], background_prob, ca_prob, n_prob, c_prob]
    test_data = [
        [[1.0, 2.0, 3.0], 0.1, 0.8, 0.05, 0.05],
        [[1.1, 2.1, 3.1], 0.2, 0.7, 0.05, 0.05],
        [[1.2, 1.9, 2.9], 0.15, 0.75, 0.05, 0.05],
        [[5.0, 6.0, 7.0], 0.3, 0.05, 0.6, 0.05],
        [[5.1, 6.1, 7.1], 0.25, 0.05, 0.65, 0.05],
        [[10.0, 11.0, 12.0], 0.4, 0.05, 0.05, 0.5],
        [[10.1, 11.1, 12.1], 0.35, 0.05, 0.05, 0.55],
    ]
    
    # Extract CA, N, C points with probability threshold
    prob_threshold = 0.4
    ca_points = []
    n_points = []
    c_points = []
    
    for data in test_data:
        coords = data[0]
        _, ca_prob, n_prob, c_prob = data[1:]
        
        if ca_prob >= prob_threshold:
            ca_points.append(WeightedPoint(coords[0], coords[1], coords[2], ca_prob))
        if n_prob >= prob_threshold:
            n_points.append(WeightedPoint(coords[0], coords[1], coords[2], n_prob))
        if c_prob >= prob_threshold:
            c_points.append(WeightedPoint(coords[0], coords[1], coords[2], c_prob))
    
    print(f"CA points above threshold: {len(ca_points)}")
    print(f"N points above threshold: {len(n_points)}")
    print(f"C points above threshold: {len(c_points)}")
    
    # Test clustering on each atom type
    clustering = AdvancedClusteringMethods()
    
    for atom_type, points in [('CA', ca_points), ('N', n_points), ('C', c_points)]:
        if not points:
            print(f"No {atom_type} points to cluster")
            continue
        
        print(f"\nClustering {atom_type} atoms:")
        result = clustering.cluster_points(points, 'adaptive_dbscan')
        print(f"  Clusters: {result.n_clusters}")
        print(f"  Centroids: {result.centroids}")
        
        # Test clustering report
        report = clustering.get_clustering_report(result)
        print(f"  Report:\n{report}")


def create_test_output_files():
    """Create test output files to verify MRC writing functionality."""
    print("\nCreating test output files...")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_probabilities_atom.txt', delete=False) as f:
        # Write test data in the expected format
        test_data = [
            [[1.0, 2.0, 3.0], 0.1, 0.8, 0.05, 0.05],
            [[1.1, 2.1, 3.1], 0.2, 0.7, 0.05, 0.05],
            [[1.2, 1.9, 2.9], 0.15, 0.75, 0.05, 0.05],
            [[5.0, 6.0, 7.0], 0.3, 0.05, 0.6, 0.05],
            [[5.1, 6.1, 7.1], 0.25, 0.05, 0.65, 0.05],
            [[10.0, 11.0, 12.0], 0.4, 0.05, 0.05, 0.5],
            [[10.1, 11.1, 12.1], 0.35, 0.05, 0.05, 0.55],
        ]
        
        for data in test_data:
            f.write(str(data) + '\n')
        
        prob_file = f.name
    
    print(f"Created test probability file: {prob_file}")
    print("You can test the clustering script with:")
    print(f"python utils/cluster_cn_predicted_map.py {prob_file} <reference_map.mrc> <output.mrc> --clustering_method adaptive_dbscan")
    
    return prob_file


def main():
    """Main test function."""
    print("Advanced Clustering Methods Test Suite")
    print("=" * 60)
    
    # Run tests
    results = test_clustering_methods()
    test_real_data_format()
    test_file = create_test_output_files()
    
    print(f"\nTest completed successfully!")
    print(f"Test probability file created: {test_file}")
    print(f"You can clean up the test file with: rm {test_file}")


if __name__ == "__main__":
    main()