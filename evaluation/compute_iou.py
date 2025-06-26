import argparse

import mrcfile
import numpy as np
from scipy import ndimage as ndi


def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask by ``radius`` voxels in all directions."""
    if radius <= 0:
        return mask
    struct = np.ones((radius * 2 + 1,) * 3, dtype=bool)
    return ndi.binary_dilation(mask, structure=struct)


def compute_iou(
    prediction_file: str,
    label_file: str,
    radius: int = 0,
) -> None:
    """Compute per-class and average IoU between two MRC files.

    If ``radius`` is greater than zero, each labeled voxel is expanded into a
    cubic bounding box with side ``2*radius+1`` before computing IoU.
    """
    with mrcfile.open(prediction_file, mode='r') as mrc:
        pred = mrc.data.astype(np.int64)
    with mrcfile.open(label_file, mode='r') as mrc:
        label = mrc.data.astype(np.int64)

    if pred.shape != label.shape:
        raise ValueError('Prediction and label volumes must have the same shape')

    class_ids = np.union1d(np.unique(pred), np.unique(label))
    iou_scores = {}

    for cid in class_ids:
        pred_mask = pred == cid
        label_mask = label == cid
        if radius > 0:
            pred_mask = _expand_mask(pred_mask, radius)
            label_mask = _expand_mask(label_mask, radius)
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        if union == 0:
            iou_scores[cid] = float('nan')
        else:
            iou_scores[cid] = intersection / union

    for cid in sorted(iou_scores):
        val = iou_scores[cid]
        if np.isnan(val):
            print(f'class {cid}: IoU = N/A (no voxels)')
        else:
            print(f'class {cid}: IoU = {val:.4f}')

    avg_iou = np.nanmean(list(iou_scores.values()))
    print(f'Average IoU: {avg_iou:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute IoU between predicted and ground truth MRC files')
    parser.add_argument('prediction', help='Predicted MRC volume')
    parser.add_argument('label', help='Ground-truth label MRC volume')
    parser.add_argument(
        '--radius', type=int, default=0,
        help='radius of cubic bounding box around each voxel')
    args = parser.parse_args()
    compute_iou(args.prediction, args.label, args.radius)


if __name__ == '__main__':
    main()
