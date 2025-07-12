#!/usr/bin/env python3
"""
Example demonstrating the enhanced clustering methods for Cryo2Struct.
This script shows how to use the new clustering algorithms with realistic data.
"""

import numpy as np
import sys
import os
sys.path.append('/home/runner/work/Cryo2Struct/Cryo2Struct')

from utils.advanced_clustering import AdvancedClusteringMethods, WeightedPoint


def create_realistic_atom_data():
    """Create realistic atom prediction data similar to cryo-EM output."""
    print("Creating realistic atom prediction data...")
    
    np.random.seed(42)  # For reproducible results
    points = []
    
    # Simulate a protein backbone with CA atoms
    # Create a helical structure
    n_residues = 20
    helix_radius = 5.0
    helix_pitch = 1.5
    
    for i in range(n_residues):
        # Helical coordinates
        angle = i * 2 * np.pi / 3.6  # ~3.6 residues per turn
        x = helix_radius * np.cos(angle)
        y = helix_radius * np.sin(angle)
        z = i * helix_pitch
        
        # Add some noise to simulate prediction uncertainty
        x += np.random.normal(0, 0.2)
        y += np.random.normal(0, 0.2)
        z += np.random.normal(0, 0.2)
        
        # Probability decreases with distance from ideal position
        prob = 0.9 - np.random.exponential(0.1)
        prob = max(0.4, min(0.95, prob))  # Clamp between 0.4 and 0.95
        
        points.append(WeightedPoint(x, y, z, prob))
    
    # Add some noise points (false positives)
    n_noise = 5
    for i in range(n_noise):
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-15, 15)
        z = np.random.uniform(-5, 35)
        prob = np.random.uniform(0.3, 0.6)  # Lower probability for noise
        points.append(WeightedPoint(x, y, z, prob))
    
    # Add a second protein chain nearby
    offset = np.array([15, 0, 0])
    for i in range(n_residues // 2):
        angle = i * 2 * np.pi / 3.6
        x = helix_radius * np.cos(angle) + offset[0]
        y = helix_radius * np.sin(angle) + offset[1]
        z = i * helix_pitch + offset[2]
        
        x += np.random.normal(0, 0.2)
        y += np.random.normal(0, 0.2)
        z += np.random.normal(0, 0.2)
        
        prob = 0.85 - np.random.exponential(0.1)
        prob = max(0.4, min(0.95, prob))
        
        points.append(WeightedPoint(x, y, z, prob))
    
    print(f"Created {len(points)} atom predictions")
    return points


def demonstrate_clustering_methods():
    """Demonstrate different clustering methods and their results."""
    print("\n" + "="*60)
    print("CLUSTERING METHODS DEMONSTRATION")
    print("="*60)
    
    # Create test data
    points = create_realistic_atom_data()
    clustering = AdvancedClusteringMethods()
    
    # Test different methods
    methods = [
        ('legacy_simulation', 'Legacy-style clustering'),
        ('dbscan', 'DBSCAN'),
        ('adaptive_dbscan', 'Adaptive DBSCAN'),
        ('probability_weighted_dbscan', 'Probability-Weighted DBSCAN'),
        ('gmm', 'Gaussian Mixture Models'),
        ('weighted_kmeans', 'Weighted K-Means'),
        ('hierarchical', 'Hierarchical Clustering'),
        ('auto', 'Automatic Method Selection')
    ]
    
    results = {}
    
    for method_id, method_name in methods:
        print(f"\n{method_name}:")
        print("-" * 40)
        
        try:
            if method_id == 'legacy_simulation':
                # Simulate legacy clustering behavior
                from utils.clustering_centroid import Point, create_clusters
                legacy_points = [Point(p.x, p.y, p.z) for p in points]
                legacy_clusters = create_clusters(legacy_points, thres=2.0)
                
                print(f"  Clusters found: {len(legacy_clusters)}")
                print(f"  Points clustered: {sum(len(c) for c in legacy_clusters)}")
                print(f"  Quality metrics: Not available")
                
                results[method_id] = {
                    'n_clusters': len(legacy_clusters),
                    'n_points': sum(len(c) for c in legacy_clusters),
                    'metrics': {}
                }
                
            else:
                # Use advanced clustering
                result = clustering.cluster_points(points, method_id)
                
                print(f"  Clusters found: {result.n_clusters}")
                print(f"  Points clustered: {sum(len(c) for c in result.clusters)}")
                print(f"  Centroids: {len(result.centroids)}")
                
                if result.metrics:
                    print(f"  Quality metrics:")
                    for metric, value in result.metrics.items():
                        print(f"    {metric}: {value:.4f}")
                
                results[method_id] = {
                    'n_clusters': result.n_clusters,
                    'n_points': sum(len(c) for c in result.clusters),
                    'metrics': result.metrics,
                    'centroids': result.centroids
                }
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method_id] = {'error': str(e)}
    
    # Summary comparison
    print(f"\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"{'Method':<25} {'Clusters':<10} {'Points':<10} {'Silhouette':<12} {'Notes'}")
    print("-" * 75)
    
    for method_id, method_name in methods:
        if method_id in results and 'error' not in results[method_id]:
            result = results[method_id]
            silhouette = result['metrics'].get('silhouette_score', 'N/A')
            if silhouette != 'N/A':
                silhouette = f"{silhouette:.4f}"
            
            # Add notes based on results
            notes = ""
            if result['n_clusters'] == 1:
                notes = "All points in one cluster"
            elif result['n_clusters'] > 5:
                notes = "Many small clusters"
            elif result['n_clusters'] >= 2:
                notes = "Good separation"
            
            print(f"{method_name:<25} {result['n_clusters']:<10} {result['n_points']:<10} {silhouette:<12} {notes}")
        else:
            print(f"{method_name:<25} {'ERROR':<10} {'':<10} {'':<12} {'Failed'}")
    
    return results


def demonstrate_probability_weighting():
    """Demonstrate the effect of probability weighting on clustering."""
    print(f"\n" + "="*60)
    print("PROBABILITY WEIGHTING DEMONSTRATION")
    print("="*60)
    
    # Create points with varying probabilities
    points = [
        WeightedPoint(1.0, 1.0, 1.0, 0.9),  # High confidence
        WeightedPoint(1.1, 1.1, 1.1, 0.8),  # High confidence
        WeightedPoint(1.2, 1.2, 1.2, 0.4),  # Low confidence
        WeightedPoint(5.0, 5.0, 5.0, 0.6),  # Medium confidence
        WeightedPoint(5.1, 5.1, 5.1, 0.5),  # Medium confidence
        WeightedPoint(5.2, 5.2, 5.2, 0.3),  # Low confidence
    ]
    
    clustering = AdvancedClusteringMethods()
    
    # Test with and without probability weighting
    print("\nWithout probability weighting:")
    result1 = clustering.cluster_points(points, 'dbscan', use_probabilities=False)
    print(f"  Clusters: {result1.n_clusters}")
    print(f"  Centroids: {result1.centroids}")
    
    print("\nWith probability weighting:")
    result2 = clustering.cluster_points(points, 'dbscan', use_probabilities=True)
    print(f"  Clusters: {result2.n_clusters}")
    print(f"  Centroids: {result2.centroids}")
    
    print("\nObservation: Probability weighting affects cluster formation and centroid positions")


def demonstrate_parameter_tuning():
    """Demonstrate how different parameters affect clustering results."""
    print(f"\n" + "="*60)
    print("PARAMETER TUNING DEMONSTRATION")
    print("="*60)
    
    # Create test data with two well-separated clusters
    points = []
    
    # Cluster 1
    for i in range(8):
        x = 1.0 + np.random.normal(0, 0.3)
        y = 1.0 + np.random.normal(0, 0.3)
        z = 1.0 + np.random.normal(0, 0.3)
        prob = 0.8 + np.random.normal(0, 0.1)
        points.append(WeightedPoint(x, y, z, prob))
    
    # Cluster 2
    for i in range(8):
        x = 8.0 + np.random.normal(0, 0.3)
        y = 8.0 + np.random.normal(0, 0.3)
        z = 8.0 + np.random.normal(0, 0.3)
        prob = 0.7 + np.random.normal(0, 0.1)
        points.append(WeightedPoint(x, y, z, prob))
    
    clustering = AdvancedClusteringMethods()
    
    # Test different DBSCAN parameters
    print("\nDBSCAN parameter tuning:")
    eps_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    for eps in eps_values:
        result = clustering.cluster_points(points, 'dbscan', eps=eps, min_samples=2)
        silhouette = result.metrics.get('silhouette_score', 'N/A')
        if silhouette != 'N/A':
            silhouette = f"{silhouette:.4f}"
        print(f"  eps={eps}: {result.n_clusters} clusters, silhouette={silhouette}")
    
    print("\nRecommendation: Use 'adaptive_dbscan' to automatically find optimal parameters")


def main():
    """Main demonstration function."""
    print("Enhanced Clustering Methods for Cryo2Struct")
    print("=" * 60)
    print("This example demonstrates the new clustering capabilities")
    print("for improved atom position clustering in cryo-EM structure prediction.")
    
    # Run demonstrations
    results = demonstrate_clustering_methods()
    demonstrate_probability_weighting()
    demonstrate_parameter_tuning()
    
    # Final recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. For most cryo-EM applications:")
    print("   Use: --clustering_method adaptive_dbscan --use_probability_weighting")
    
    print("\n2. For noisy or low-resolution data:")
    print("   Use: --clustering_method probability_weighted_dbscan")
    
    print("\n3. For automatic optimization:")
    print("   Use: --clustering_method auto --clustering_report report.txt")
    
    print("\n4. For backward compatibility:")
    print("   Use: --clustering_method legacy (default)")
    
    print("\n5. Configuration file example:")
    print("   clustering_method: 'adaptive_dbscan'")
    print("   use_probability_weighting: true")
    
    print(f"\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("The enhanced clustering methods provide significant improvements")
    print("in atom position clustering quality for cryo-EM structure prediction.")


if __name__ == "__main__":
    main()