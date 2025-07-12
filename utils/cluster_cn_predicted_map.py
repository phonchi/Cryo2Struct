import argparse
import ast
import math

import mrcfile
import numpy as np
from scipy.spatial import cKDTree


from utils.clustering_centroid import Point, create_clusters
from utils.advanced_clustering import AdvancedClusteringMethods, WeightedPoint


class WeightedPoint(Point):
    """Point with an associated probability for clustering."""

    def __init__(self, x: float, y: float, z: float, prob: float):
        super().__init__(x, y, z)
        self.prob = prob
    
    def to_array(self):
        """Convert to numpy array for clustering algorithms."""
        return np.array([self.x, self.y, self.z])


def advanced_clustering(points, method='dbscan', **kwargs):
    """
    Apply advanced clustering methods to points.
    
    Args:
        points: List of WeightedPoint objects
        method: Clustering method ('dbscan', 'gmm', 'weighted_kmeans', 'hierarchical', 
                'adaptive_dbscan', 'probability_weighted_dbscan', 'auto')
        **kwargs: Method-specific parameters
    
    Returns:
        List of clusters (each cluster is a list of WeightedPoint objects)
    """
    if not points:
        return []
    
    clustering = AdvancedClusteringMethods()
    
    if method == 'auto':
        result = clustering.select_best_method(points, **kwargs)
    else:
        result = clustering.cluster_points(points, method, **kwargs)
    
    return result.clusters


def get_index(cord: float, origin: float, voxel: float) -> int:
    """Convert a coordinate to its voxel index."""
    return int(math.floor((cord - origin) / voxel))


def parse_probabilities(prob_file: str, prob_threshold: float):
    """Return lists of WeightedPoints for CA, N and C above the threshold."""
    ca_points, n_points, c_points = [], [], []
    with open(prob_file, "r") as f:
        for line in f:
            vals = ast.literal_eval(line)
            x, y, z = vals[0]
            _, ca_p, n_p, c_p = vals[1:]
            if ca_p >= prob_threshold:
                ca_points.append(WeightedPoint(x, y, z, ca_p))
            if n_p >= prob_threshold:
                n_points.append(WeightedPoint(x, y, z, n_p))
            if c_p >= prob_threshold:
                c_points.append(WeightedPoint(x, y, z, c_p))
    return ca_points, n_points, c_points



def nms_basic(points, radius):
    """Simple NMS that iteratively suppresses neighbors within ``radius``."""
    if radius <= 0:
        return points
    points = sorted(points, key=lambda p: p.prob, reverse=True)
    kept = []
    for p in points:
        keep = True
        for q in kept:
            if math.dist((p.x, p.y, p.z), (q.x, q.y, q.z)) <= radius:
                keep = False
                break
        if keep:
            kept.append(p)
    return kept


def nms_kdtree(points, radius):
    """Efficient NMS using a cKDTree for neighbor queries."""
    if radius <= 0 or not points:
        return points

    coords = np.array([(p.x, p.y, p.z) for p in points])
    probs = np.array([p.prob for p in points])
    order = np.argsort(-probs)
    tree = cKDTree(coords)
    suppressed = np.zeros(len(points), dtype=bool)
    kept = []

    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(points[idx])
        neighbors = tree.query_ball_point(coords[idx], r=radius)
        suppressed[neighbors] = True

    return kept


def centroids_from_clusters(clusters, use_probability_weighting=True):
    """
    Return (x, y, z, avg_prob) tuples for each cluster.
    
    Args:
        clusters: List of clusters (each cluster is a list of WeightedPoint objects)
        use_probability_weighting: Whether to use probability-weighted centroids
    """
    results = []
    for cluster in clusters:
        if not cluster:
            continue
        
        if use_probability_weighting:
            # Compute probability-weighted centroid
            total_weight = sum(p.prob for p in cluster)
            if total_weight == 0:
                total_weight = len(cluster)
            
            weighted_x = sum(p.x * p.prob for p in cluster) / total_weight
            weighted_y = sum(p.y * p.prob for p in cluster) / total_weight
            weighted_z = sum(p.z * p.prob for p in cluster) / total_weight
            avg_prob = sum(p.prob for p in cluster) / len(cluster)
            
            results.append((weighted_x, weighted_y, weighted_z, avg_prob))
        else:
            # Simple average centroid
            xs = [p.x for p in cluster]
            ys = [p.y for p in cluster]
            zs = [p.z for p in cluster]
            ps = [p.prob for p in cluster]
            results.append(
                (
                    sum(xs) / len(xs),
                    sum(ys) / len(ys),
                    sum(zs) / len(zs),
                    sum(ps) / len(ps),
                )
            )
    return results


def write_centroid_file(centroids, out_path):
    """Write centroid coordinates and average probability to a text file."""
    if not out_path:
        return
    with open(out_path, "w") as fh:
        for x, y, z, p in centroids:
            fh.write(f"{x} {y} {z} {p}\n")


def write_mrc(ca_centroids, n_centroids, c_centroids, ref_map, out_path):
    with mrcfile.open(ref_map, mode="r") as m:
        data = np.zeros_like(m.data, dtype=np.int16)
        origin = m.header.origin  # preserve the reference origin record
        x_origin = origin["x"]
        y_origin = origin["y"]
        z_origin = origin["z"]
        x_voxel = m.voxel_size["x"]
        y_voxel = m.voxel_size["y"]
        z_voxel = m.voxel_size["z"]

    def place(points, label):
        for x, y, z, _ in points:
            iz = get_index(z, z_origin, z_voxel)
            jy = get_index(y, y_origin, y_voxel)
            kx = get_index(x, x_origin, x_voxel)
            if 0 <= iz < data.shape[0] and 0 <= jy < data.shape[1] and 0 <= kx < data.shape[2]:
                data[iz, jy, kx] = label

    place(ca_centroids, 1)
    place(n_centroids, 2)
    place(c_centroids, 3)

    with mrcfile.new(out_path, overwrite=True) as m:
        m.set_data(data.astype(np.float32))
        m.voxel_size = x_voxel
        m.header.origin = origin


def main():
    parser = argparse.ArgumentParser(
        description="Cluster atom predictions or apply NMS to create a labeled map"
    )
    parser.add_argument("prob_file", help="probabilities_atom.txt produced by inference")
    parser.add_argument("reference_map", help="reference MRC map for shape and metadata")
    parser.add_argument("output", help="output MRC file with suppressed atoms")
    parser.add_argument("--ca_txt", help="optional output file for CA centroids")
    parser.add_argument("--n_txt", help="optional output file for N centroids")
    parser.add_argument("--c_txt", help="optional output file for C centroids")
    parser.add_argument("--prob_threshold", type=float, default=0.4, help="minimum probability to keep a voxel")
    parser.add_argument("--cluster_threshold", type=float, default=2.0, help="distance threshold for clustering")
    parser.add_argument("--nms_radius", type=float, default=0.0, help="apply non-maximum suppression with this radius")
    parser.add_argument(
        "--nms_method",
        choices=["basic", "kdtree"],
        default="basic",
        help="NMS implementation to use when --nms_radius > 0",
    )
    parser.add_argument(
        "--clustering_method",
        choices=["legacy", "dbscan", "gmm", "weighted_kmeans", "hierarchical", "adaptive_dbscan", "probability_weighted_dbscan", "auto"],
        default="legacy",
        help="clustering method to use",
    )
    parser.add_argument("--dbscan_eps", type=float, default=2.0, help="DBSCAN epsilon parameter")
    parser.add_argument("--dbscan_min_samples", type=int, default=2, help="DBSCAN minimum samples parameter")
    parser.add_argument("--gmm_components", type=int, default=None, help="GMM number of components (auto if not set)")
    parser.add_argument("--kmeans_clusters", type=int, default=None, help="K-means number of clusters (auto if not set)")
    parser.add_argument("--hierarchical_clusters", type=int, default=None, help="Hierarchical clustering number of clusters (auto if not set)")
    parser.add_argument("--use_probability_weighting", action="store_true", help="use probability-weighted centroids")
    parser.add_argument("--clustering_report", help="optional file to save clustering quality report")
    args = parser.parse_args()

    ca_pts, n_pts, c_pts = parse_probabilities(args.prob_file, args.prob_threshold)

    if args.nms_radius > 0:
        nms_func = nms_kdtree if args.nms_method == "kdtree" else nms_basic
        ca_pts = nms_func(ca_pts, args.nms_radius)
        n_pts = nms_func(n_pts, args.nms_radius)
        c_pts = nms_func(c_pts, args.nms_radius)

    # Apply clustering
    if args.clustering_method == "legacy":
        # Use original clustering method for backward compatibility
        ca_clusters = create_clusters(ca_pts, args.cluster_threshold)
        n_clusters = create_clusters(n_pts, args.cluster_threshold)
        c_clusters = create_clusters(c_pts, args.cluster_threshold)
    else:
        # Use advanced clustering methods
        clustering_kwargs = {}
        
        if args.clustering_method == "dbscan":
            clustering_kwargs = {
                'eps': args.dbscan_eps,
                'min_samples': args.dbscan_min_samples,
                'use_probabilities': True
            }
        elif args.clustering_method == "gmm":
            clustering_kwargs = {
                'n_components': args.gmm_components
            }
        elif args.clustering_method == "weighted_kmeans":
            clustering_kwargs = {
                'n_clusters': args.kmeans_clusters
            }
        elif args.clustering_method == "hierarchical":
            clustering_kwargs = {
                'n_clusters': args.hierarchical_clusters
            }
        elif args.clustering_method == "adaptive_dbscan":
            clustering_kwargs = {
                'k': 3,
                'percentile': 90
            }
        elif args.clustering_method == "probability_weighted_dbscan":
            clustering_kwargs = {
                'base_eps': args.dbscan_eps,
                'prob_weight': 0.5,
                'min_samples': args.dbscan_min_samples
            }
        
        ca_clusters = advanced_clustering(ca_pts, method=args.clustering_method, **clustering_kwargs)
        n_clusters = advanced_clustering(n_pts, method=args.clustering_method, **clustering_kwargs)
        c_clusters = advanced_clustering(c_pts, method=args.clustering_method, **clustering_kwargs)
        
        # Generate clustering report if requested
        if args.clustering_report:
            clustering = AdvancedClusteringMethods()
            ca_result = clustering.cluster_points(ca_pts, args.clustering_method, **clustering_kwargs)
            n_result = clustering.cluster_points(n_pts, args.clustering_method, **clustering_kwargs)
            c_result = clustering.cluster_points(c_pts, args.clustering_method, **clustering_kwargs)
            
            with open(args.clustering_report, 'w') as f:
                f.write("=== Clustering Quality Report ===\n\n")
                f.write("CA (Carbon Alpha) Atoms:\n")
                f.write(clustering.get_clustering_report(ca_result))
                f.write("\n\nN (Nitrogen) Atoms:\n")
                f.write(clustering.get_clustering_report(n_result))
                f.write("\n\nC (Carbon) Atoms:\n")
                f.write(clustering.get_clustering_report(c_result))
                f.write("\n")

    ca_centroids = centroids_from_clusters(ca_clusters, args.use_probability_weighting)
    n_centroids = centroids_from_clusters(n_clusters, args.use_probability_weighting)
    c_centroids = centroids_from_clusters(c_clusters, args.use_probability_weighting)

    write_mrc(ca_centroids, n_centroids, c_centroids, args.reference_map, args.output)
    write_centroid_file(ca_centroids, args.ca_txt)
    write_centroid_file(n_centroids, args.n_txt)
    write_centroid_file(c_centroids, args.c_txt)


if __name__ == "__main__":
    main()