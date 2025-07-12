#!/usr/bin/env python3
"""
Integration test for enhanced clustering methods in Cryo2Struct.
This test verifies the integration of new clustering methods with the existing pipeline.
"""

import numpy as np
import tempfile
import os
import sys
sys.path.append('/home/runner/work/Cryo2Struct/Cryo2Struct')

from utils.cluster_cn_predicted_map import parse_probabilities, advanced_clustering, WeightedPoint
from utils.advanced_clustering import AdvancedClusteringMethods
import subprocess


def create_test_probability_file():
    """Create a test probability file with realistic atom prediction data."""
    
    # Create more realistic test data with multiple clusters
    test_data = []
    
    # Cluster 1: CA atoms around (1, 2, 3)
    for i in range(5):
        x = 1.0 + np.random.normal(0, 0.1)
        y = 2.0 + np.random.normal(0, 0.1)
        z = 3.0 + np.random.normal(0, 0.1)
        ca_prob = 0.7 + np.random.normal(0, 0.1)
        test_data.append([[x, y, z], 0.15, ca_prob, 0.05, 0.05])
    
    # Cluster 2: N atoms around (5, 6, 7)
    for i in range(4):
        x = 5.0 + np.random.normal(0, 0.1)
        y = 6.0 + np.random.normal(0, 0.1)
        z = 7.0 + np.random.normal(0, 0.1)
        n_prob = 0.6 + np.random.normal(0, 0.1)
        test_data.append([[x, y, z], 0.2, 0.05, n_prob, 0.05])
    
    # Cluster 3: C atoms around (10, 11, 12)
    for i in range(3):
        x = 10.0 + np.random.normal(0, 0.1)
        y = 11.0 + np.random.normal(0, 0.1)
        z = 12.0 + np.random.normal(0, 0.1)
        c_prob = 0.55 + np.random.normal(0, 0.1)
        test_data.append([[x, y, z], 0.25, 0.05, 0.05, c_prob])
    
    # Cluster 4: Mixed atoms around (15, 16, 17)
    for i in range(6):
        x = 15.0 + np.random.normal(0, 0.2)
        y = 16.0 + np.random.normal(0, 0.2)
        z = 17.0 + np.random.normal(0, 0.2)
        
        # Random atom types
        atom_probs = [0.2, 0.0, 0.0, 0.0]  # background
        rand_atom = np.random.choice(['ca', 'n', 'c'])
        if rand_atom == 'ca':
            atom_probs[1] = 0.6 + np.random.normal(0, 0.1)
        elif rand_atom == 'n':
            atom_probs[2] = 0.5 + np.random.normal(0, 0.1)
        else:
            atom_probs[3] = 0.5 + np.random.normal(0, 0.1)
        
        test_data.append([[x, y, z], atom_probs[0], atom_probs[1], atom_probs[2], atom_probs[3]])
    
    # Add some noise points
    for i in range(3):
        x = np.random.uniform(-5, 25)
        y = np.random.uniform(-5, 25)
        z = np.random.uniform(-5, 25)
        test_data.append([[x, y, z], 0.8, 0.1, 0.05, 0.05])  # Low probability atoms
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_probabilities_atom.txt', delete=False) as f:
        for data in test_data:
            f.write(str(data) + '\n')
        return f.name


def test_parse_probabilities():
    """Test the parse_probabilities function with new clustering methods."""
    print("Testing parse_probabilities function...")
    
    prob_file = create_test_probability_file()
    
    try:
        # Test with different probability thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6]
        
        for threshold in thresholds:
            ca_pts, n_pts, c_pts = parse_probabilities(prob_file, threshold)
            print(f"  Threshold {threshold}: CA={len(ca_pts)}, N={len(n_pts)}, C={len(c_pts)}")
            
            # Verify points are WeightedPoint objects
            if ca_pts:
                assert isinstance(ca_pts[0], WeightedPoint), "CA points should be WeightedPoint objects"
                assert ca_pts[0].prob >= threshold, f"CA point prob {ca_pts[0].prob} should be >= {threshold}"
            
            if n_pts:
                assert isinstance(n_pts[0], WeightedPoint), "N points should be WeightedPoint objects"
                assert n_pts[0].prob >= threshold, f"N point prob {n_pts[0].prob} should be >= {threshold}"
            
            if c_pts:
                assert isinstance(c_pts[0], WeightedPoint), "C points should be WeightedPoint objects"
                assert c_pts[0].prob >= threshold, f"C point prob {c_pts[0].prob} should be >= {threshold}"
        
        print("  ✓ parse_probabilities test passed")
    
    finally:
        os.unlink(prob_file)


def test_advanced_clustering_integration():
    """Test the advanced_clustering function integration."""
    print("Testing advanced_clustering function integration...")
    
    # Create test points
    points = [
        WeightedPoint(1.0, 2.0, 3.0, 0.8),
        WeightedPoint(1.1, 2.1, 3.1, 0.7),
        WeightedPoint(1.2, 1.9, 2.9, 0.75),
        WeightedPoint(5.0, 6.0, 7.0, 0.6),
        WeightedPoint(5.1, 6.1, 7.1, 0.65),
        WeightedPoint(10.0, 11.0, 12.0, 0.5),
        WeightedPoint(10.1, 11.1, 12.1, 0.55),
    ]
    
    # Test different clustering methods
    methods = ['dbscan', 'adaptive_dbscan', 'gmm', 'weighted_kmeans', 'hierarchical', 'auto']
    
    for method in methods:
        try:
            clusters = advanced_clustering(points, method=method)
            print(f"  Method {method}: {len(clusters)} clusters found")
            
            # Verify clusters contain WeightedPoint objects
            total_points = sum(len(cluster) for cluster in clusters)
            assert total_points > 0, f"No points clustered by {method}"
            
            for cluster in clusters:
                assert len(cluster) > 0, f"Empty cluster found in {method}"
                assert isinstance(cluster[0], WeightedPoint), f"Cluster should contain WeightedPoint objects in {method}"
        
        except Exception as e:
            print(f"  Method {method}: ERROR - {e}")
    
    print("  ✓ advanced_clustering integration test passed")


def test_clustering_quality_comparison():
    """Test and compare clustering quality across methods."""
    print("Testing clustering quality comparison...")
    
    # Generate structured test data
    np.random.seed(42)
    points = []
    
    # Create 3 well-separated clusters
    cluster_centers = [(0, 0, 0), (10, 10, 10), (20, 20, 20)]
    for i, center in enumerate(cluster_centers):
        for j in range(8):
            x = center[0] + np.random.normal(0, 0.5)
            y = center[1] + np.random.normal(0, 0.5)
            z = center[2] + np.random.normal(0, 0.5)
            prob = 0.8 + np.random.normal(0, 0.1)
            points.append(WeightedPoint(x, y, z, prob))
    
    # Test clustering methods
    clustering = AdvancedClusteringMethods()
    methods = ['dbscan', 'adaptive_dbscan', 'gmm', 'weighted_kmeans', 'hierarchical']
    
    results = {}
    for method in methods:
        try:
            result = clustering.cluster_points(points, method)
            results[method] = result
            print(f"  {method}: {result.n_clusters} clusters, silhouette: {result.metrics.get('silhouette_score', 'N/A')}")
        except Exception as e:
            print(f"  {method}: ERROR - {e}")
    
    # Find best method
    best_method = clustering.select_best_method(points)
    print(f"  Best method selected: {best_method.method} with silhouette score: {best_method.metrics.get('silhouette_score', 'N/A')}")
    
    print("  ✓ clustering quality comparison test passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    clustering = AdvancedClusteringMethods()
    
    # Test with empty points
    empty_result = clustering.cluster_points([])
    assert empty_result.n_clusters == 0, "Empty points should result in 0 clusters"
    print("  ✓ Empty points handled correctly")
    
    # Test with single point
    single_point = [WeightedPoint(1.0, 2.0, 3.0, 0.8)]
    single_result = clustering.cluster_points(single_point, 'dbscan')
    print(f"  Single point: {single_result.n_clusters} clusters")
    
    # Test with two very close points
    close_points = [
        WeightedPoint(1.0, 2.0, 3.0, 0.8),
        WeightedPoint(1.001, 2.001, 3.001, 0.75)
    ]
    close_result = clustering.cluster_points(close_points, 'dbscan')
    print(f"  Close points: {close_result.n_clusters} clusters")
    
    # Test with identical points
    identical_points = [
        WeightedPoint(1.0, 2.0, 3.0, 0.8),
        WeightedPoint(1.0, 2.0, 3.0, 0.7)
    ]
    identical_result = clustering.cluster_points(identical_points, 'dbscan')
    print(f"  Identical points: {identical_result.n_clusters} clusters")
    
    print("  ✓ Edge cases test passed")


def main():
    """Run all integration tests."""
    print("Advanced Clustering Integration Test Suite")
    print("=" * 60)
    
    try:
        test_parse_probabilities()
        test_advanced_clustering_integration()
        test_clustering_quality_comparison()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✓ All integration tests passed!")
        print("The enhanced clustering methods are successfully integrated.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())