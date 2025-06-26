import argparse
import ast
import math

import mrcfile
import numpy as np
from scipy.spatial import cKDTree

from utils.clustering_centroid import Point, create_clusters


class WeightedPoint(Point):
    """Point with an associated probability for clustering."""

    def __init__(self, x: float, y: float, z: float, prob: float):
        super().__init__(x, y, z)
        self.prob = prob


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


def centroids_from_clusters(clusters):
    """Return (x, y, z, avg_prob) tuples for each cluster."""
    results = []
    for cluster in clusters:
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
    parser = argparse.ArgumentParser(description="Cluster atom predictions and create a labeled map")
    parser.add_argument("prob_file", help="probabilities_atom.txt produced by inference")
    parser.add_argument("reference_map", help="reference MRC map for shape and metadata")
    parser.add_argument("output", help="output MRC file with clustered atoms")
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
    args = parser.parse_args()

    ca_pts, n_pts, c_pts = parse_probabilities(args.prob_file, args.prob_threshold)

    if args.nms_radius > 0:
        nms_func = nms_kdtree if args.nms_method == "kdtree" else nms_basic
        ca_pts = nms_func(ca_pts, args.nms_radius)
        n_pts = nms_func(n_pts, args.nms_radius)
        c_pts = nms_func(c_pts, args.nms_radius)

    ca_clusters = create_clusters(ca_pts, args.cluster_threshold)
    n_clusters = create_clusters(n_pts, args.cluster_threshold)
    c_clusters = create_clusters(c_pts, args.cluster_threshold)

    ca_centroids = centroids_from_clusters(ca_clusters)
    n_centroids = centroids_from_clusters(n_clusters)
    c_centroids = centroids_from_clusters(c_clusters)

    write_mrc(ca_centroids, n_centroids, c_centroids, args.reference_map, args.output)
    write_centroid_file(ca_centroids, args.ca_txt)
    write_centroid_file(n_centroids, args.n_txt)
    write_centroid_file(c_centroids, args.c_txt)


if __name__ == "__main__":
    main()