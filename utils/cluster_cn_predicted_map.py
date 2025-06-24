import argparse
import ast
import math

import mrcfile
import numpy as np

from utils.clustering_centroid import Point, create_clusters


def get_index(cord: float, origin: float, voxel: float) -> int:
    """Convert a coordinate to its voxel index."""
    return int(math.floor((cord - origin) / voxel))


def parse_probabilities(prob_file: str, prob_threshold: float):
    """Return separate lists of Points for CA, N and C above the threshold."""
    ca_points, n_points, c_points = [], [], []
    with open(prob_file, "r") as f:
        for line in f:
            vals = ast.literal_eval(line)
            x, y, z = vals[0]
            _, ca_p, n_p, c_p = vals[1:]
            if ca_p >= prob_threshold:
                ca_points.append(Point(x, y, z))
            if n_p >= prob_threshold:
                n_points.append(Point(x, y, z))
            if c_p >= prob_threshold:
                c_points.append(Point(x, y, z))
    return ca_points, n_points, c_points


def centroids_from_clusters(clusters):
    """Compute centroid coordinate for each cluster."""
    results = []
    for cluster in clusters:
        xs = [p.x for p in cluster]
        ys = [p.y for p in cluster]
        zs = [p.z for p in cluster]
        results.append((sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)))
    return results


def write_mrc(ca_centroids, n_centroids, c_centroids, ref_map, out_path):
    with mrcfile.open(ref_map, mode="r") as m:
        data = np.zeros_like(m.data, dtype=np.int16)
        x_origin = m.header.origin["x"]
        y_origin = m.header.origin["y"]
        z_origin = m.header.origin["z"]
        x_voxel = m.voxel_size["x"]
        y_voxel = m.voxel_size["y"]
        z_voxel = m.voxel_size["z"]

    def place(points, label):
        for x, y, z in points:
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
        m.voxel_size = (x_voxel, y_voxel, z_voxel)
        m.header.origin = {"x": x_origin, "y": y_origin, "z": z_origin}


def main():
    parser = argparse.ArgumentParser(description="Cluster atom predictions and create a labeled map")
    parser.add_argument("prob_file", help="probabilities_atom.txt produced by inference")
    parser.add_argument("reference_map", help="reference MRC map for shape and metadata")
    parser.add_argument("output", help="output MRC file with clustered atoms")
    parser.add_argument("--prob_threshold", type=float, default=0.4, help="minimum probability to keep a voxel")
    parser.add_argument("--cluster_threshold", type=float, default=2.0, help="distance threshold for clustering")
    args = parser.parse_args()

    ca_pts, n_pts, c_pts = parse_probabilities(args.prob_file, args.prob_threshold)
    ca_clusters = create_clusters(ca_pts, args.cluster_threshold)
    n_clusters = create_clusters(n_pts, args.cluster_threshold)
    c_clusters = create_clusters(c_pts, args.cluster_threshold)

    ca_centroids = centroids_from_clusters(ca_clusters)
    n_centroids = centroids_from_clusters(n_clusters)
    c_centroids = centroids_from_clusters(c_clusters)

    write_mrc(ca_centroids, n_centroids, c_centroids, args.reference_map, args.output)


if __name__ == "__main__":
    main()
