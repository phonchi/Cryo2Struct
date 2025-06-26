import argparse
import ast
import math

import mrcfile
import numpy as np


class WeightedPoint:
    """Simple 3D point with an associated probability."""

    def __init__(self, x: float, y: float, z: float, prob: float):
        self.x = x
        self.y = y
        self.z = z
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


def nms(points, radius):
    """Simple 3D non-maximum suppression."""
    if not points:
        return []
    points = sorted(points, key=lambda p: p.prob, reverse=True)
    selected = []
    r2 = radius * radius
    for p in points:
        keep = True
        for q in selected:
            dx = p.x - q.x
            dy = p.y - q.y
            dz = p.z - q.z
            if dx * dx + dy * dy + dz * dz <= r2:
                keep = False
                break
        if keep:
            selected.append(p)
    return [(pt.x, pt.y, pt.z, pt.prob) for pt in selected]


def cluster(points, radius):
    """Group nearby points and return their centroids with average probability."""
    pts = points[:]
    clusters = []
    r2 = radius * radius
    while pts:
        cluster_members = [pts.pop(0)]
        i = 0
        while i < len(pts):
            p = pts[i]
            c = cluster_members[0]
            dx = p.x - c.x
            dy = p.y - c.y
            dz = p.z - c.z
            if dx * dx + dy * dy + dz * dz <= r2:
                cluster_members.append(pts.pop(i))
            else:
                i += 1
        x_avg = sum(p.x for p in cluster_members) / len(cluster_members)
        y_avg = sum(p.y for p in cluster_members) / len(cluster_members)
        z_avg = sum(p.z for p in cluster_members) / len(cluster_members)
        p_avg = sum(p.prob for p in cluster_members) / len(cluster_members)
        clusters.append((x_avg, y_avg, z_avg, p_avg))
    return clusters


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
    parser.add_argument("--cluster_radius", type=float, default=2.0, help="radius for simple clustering")
    parser.add_argument("--nms_radius", type=float, default=0.0, help="if >0, perform non-max suppression with this radius")

    ca_pts, n_pts, c_pts = parse_probabilities(args.prob_file, args.prob_threshold)
    if args.nms_radius > 0:
        ca_centroids = nms(ca_pts, args.nms_radius)
        n_centroids = nms(n_pts, args.nms_radius)
        c_centroids = nms(c_pts, args.nms_radius)
    else:
        ca_centroids = cluster(ca_pts, args.cluster_radius)
        n_centroids = cluster(n_pts, args.cluster_radius)
        c_centroids = cluster(c_pts, args.cluster_radius)

    write_mrc(ca_centroids, n_centroids, c_centroids, args.reference_map, args.output)
    write_centroid_file(ca_centroids, args.ca_txt)
    write_centroid_file(n_centroids, args.n_txt)
    write_centroid_file(c_centroids, args.c_txt)


if __name__ == "__main__":
    main()
