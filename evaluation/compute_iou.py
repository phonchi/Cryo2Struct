import argparse
import numpy as np
import mrcfile


def compute_iou(prediction_file: str, label_file: str) -> None:
    """Compute per-class and average IoU between two MRC files."""
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
    args = parser.parse_args()
    compute_iou(args.prediction, args.label)


if __name__ == '__main__':
    main()
