from typing import Any, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


def mean_iou(
    true_masks: np.ndarray,
    pred_masks: np.ndarray
) -> Tuple[np.ndarray, float]:
    """ Compute mean IoU score over 10 IoU thresholds (0.5 - 0.95)

    Arguments:
        true_masks: Ground truth masks (3D array)
        pred_masks: Predicted masks (3D array)

    Returns:
        ious: IoU scores for each threshold (1D array)
        np.mean(ious): Mean IoU score

    Due to https://www.kaggle.com/wcukierski/example-metric-implementation
    """

    y_pred = np.sum((pred_masks.T*np.arange(1, len(pred_masks)+1)).T, axis=0)
    y_true = np.sum((true_masks.T*np.arange(1, len(true_masks)+1)).T, axis=0)
    #
    num_pred = y_pred.max()+1
    num_true = y_true.max()+1
    if num_pred < 1:
        num_pred = 1
        y_pred[0, 0] = 1
    # Compute intersection between all objects
    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(num_true, num_pred))[0]
    # print(num_pred)

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=num_true)[0]
    area_pred = np.histogram(y_pred, bins=num_pred)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    # print(union,union[1:, 1:])

    # Compute the intersection over union
    ious = intersection / union

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        if (tp + fp + fn) > 0:
            p = tp*1.0 / (tp + fp + fn)
        else:
            p = 0

        prec.append(p)

    return ious, np.mean(prec)


# Precision helper function
def precision_at(
    threshold: float,
    iou: np.ndarray
) -> Tuple[float, float, float]:
    """Compute precision at a given threshold on IoU.

    Arguments:
        threshold: Threshold to use for computing precision
        iou: Matrix of IoU values (2D array)

    Returns:
        tp, fp, fn: True positives, false positives, false negatives
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(
        false_positives), np.sum(false_negatives)
    return tp, fp, fn


def binarized_to_labels(m: np.ndarray) -> np.ndarray:
    """Convert a binarized mask to a labelled mask

    Arguments:
        m: Binarized mask (2D array)

    Returns:
        m: Labelled mask (2D array)
    """
    m, _ = scipy.ndimage.label(m)
    return m


def extract_individual_masks(m: np.ndarray) -> np.ndarray:
    """Extract individual masks from a labelled mask

    Arguments:
        m: Labelled mask (2D array)

    Returns:
        masks: List of individual masks (3D array)
    """
        
    
    # extract unique ids
    ids = list(np.unique(m))

    # remove background id if needed
    if 0 in ids:
        ids.remove(0)

    # create masks
    masks = [(m == id_).astype(int) for id_ in ids]

    return np.array(masks)


def calculate_the_scores(
    mask1: np.ndarray,
    mask2: np.ndarray,
    iou_thr: float = 0.5,
    borders: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[float, float, float]:
    """Calculate the scores for a given ground truth and predicted mask
    
    Arguments:
        mask1: Ground truth mask (labelled 2D array)
        mask2: Predicted mask (labelled 2D array)
        iou_thr: Intersection over union threshold (default: 0.5)
        borders: Borders of the image to evaluate (default: None)
        
    Returns:
        j: Jaccard index
        sd: Dice similarity coefficient
        jc: Jaccard index for each cell
    """

    if (borders != None):
        mask1 = mask1[borders[0]:borders[1], borders[2]:borders[3]]
        mask2 = mask2[borders[0]:borders[1], borders[2]:borders[3]]

    if (np.max(mask2) != 0):

        masks1 = extract_individual_masks(mask1)
        masks2 = extract_individual_masks(mask2)
        iou, _ = mean_iou(masks1, masks2)
        tp, fp, fn = precision_at(iou_thr, iou)
        jc = np.mean(iou[iou > iou_thr])
        j = tp / (tp + fp + fn)
        sd = 2*tp / (2*tp + fp + fn)

    elif (np.max(mask2) == 0):  # if predicted mask is emtpye; all cells are FN
        j = 0
        sd = 0
        jc = -1

    return j, sd, jc


def evaluate(
    gt_path: str,
    pred_path: str,
    iou_thr: float = 0.5,
    borders: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[float, float, float]:
    """Evaluate the segmentation performance of a predicted image

    Arguments:
        gt_path: Path to ground truth image (HDF5 file)
        pred_path: Path to predicted image (HDF5 file)
        iou_thr: Intersection over union threshold (default: 0.5)
        borders: Borders of the image to evaluate (default: None)

    Returns:
        j: Jaccard index
        sd: Dice similarity coefficient
        jc: Jaccard index for each cell
    """

    h5_gt = h5py.File(gt_path, 'r')
    masks_gt = np.array(h5_gt['FOV0']['T0'], dtype=int)
    # TODO parametrize binarized_to_labels
    # masks_gt = binarized_to_labels(masks_gt)

    h5_pred = h5py.File(pred_path, 'r')
    masks_pred = np.array(h5_pred['FOV0']['T0'], dtype=int)

    j, sd, jc = calculate_the_scores(masks_gt, masks_pred, iou_thr, borders)

    return j, sd, jc


def save_metrics(
    metrics_dict: dict,
    path: str
) -> None:
    """Save metrics to CSV file

    Numpy is used to save the metrics to a CSV file, as we did not want to add dependencies on new packages.
    The CSV file is structured as follows:
        epoch,J,SD,Jc

    Arguments:
        metrics_dict: Dictionary of average metrics (J, SD, Jc) on segmented style transferred images for each epoch
        path: Path to save CSV file
    """

    # Convert metrics_per_epoch to a structured NumPy array
    metrics_arr = np.empty(
        len(metrics_dict),
        dtype=[('epoch', int), ('J', float), ('SD', float), ('Jc', float)]
    )
    for i, (epoch, metrics) in enumerate(metrics_dict.items()):
        metrics_arr[i] = (epoch, *metrics)

    # Write to CSV
    np.savetxt(path, metrics_arr, delimiter=',',
               header='epoch,AP,SD,Jc', fmt='%d,%f,%f,%f')


def load_metrics(path: str) -> np.ndarray:
    """Load metrics from CSV file

    Arguments:
        path: Path to CSV file

    Returns:
        metrics: NumPy array of metrics
    """
    metrics = np.loadtxt(path, delimiter=',', skiprows=1)
    return metrics


def load_cycle_gan_losses(loss_log_path: str) -> dict:
    """Load Cycle GAN losses from log file

    Arguments:
        loss_log_path: Path to log file

    Returns:
        losses: Dictionary of losses
    """
    # Load cycle GAN logs
    with open(loss_log_path) as f:
        lines = f.readlines()
        lines = lines[1:]

    losses: dict = {}
    keys = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

    for key in keys:
        losses[key] = {}

    for line in lines:
        start = int(line.find(':'))
        end = int(line.find(','))
        epoch = int(line[start+1:end])
        for key in keys:
            start = int(line.find(key))
            length = len(key)+2
            if epoch not in losses[key]:
                losses[key][epoch] = []
            losses[key][epoch].append(float(line[start+length:start+length+5]))

    for key in keys:
        for epoch, values in losses[key].items():
            losses[key][epoch] = sum(losses[key][epoch])

    return losses


def plot_metrics(
    metrics_path: str,
    loss_log_path: str,
    original_domain: str,
    max_epoch: Optional[int] = None,
    save_path: str = ''
) -> None:
    """Plot metrics and losses

    Arguments:
        metrics_path: Path to metrics CSV file
        loss_log_path: Path to Cycle GAN loss log file
        original_domain: Domain of original images (A or B)
        max_epoch: Maximum epoch to plot
        save_path: Path to save plot
    """

    losses = load_cycle_gan_losses(loss_log_path)
    metrics = load_metrics(metrics_path)

    _, axs = plt.subplots(2, 1, figsize=(20, 10))
    plt.ylim([0, 1])
    axs[0].set_ylim([0, 1.05])
    axs[1].set_ylim([0, 1.05])
    if max_epoch is not None:
        axs[0].set_xlim([-2, max_epoch])
        axs[1].set_xlim([-2, max_epoch])

    graph_colors = ['#1f78b4', '#8DB220',
                    '#F8333C', '#FF813D', '#555B6E', '#00A6FB']

    # Plot Yeaz metrics
    axs[1].plot(metrics[:, 0], metrics[:, 1], linewidth=5,
                label='Patch AP metrics', color=graph_colors[3])

    # Plot CycleGAN losses
    if (original_domain == 'A'):
        keys = ['D_A', 'G_A', 'cycle_A']
    else:
        keys = ['D_B', 'G_B', 'cycle_B']
    for i, key in enumerate(keys):
        axs[0].plot(losses[key].keys(), np.array(list(losses[key].values())) /
                    max(losses[key].values()), label=key, linewidth=3, color=graph_colors[i])

    # Add title and legend
    axs[0].set_title('Cycle GAN losses', fontsize=20)
    axs[1].set_title('Yeaz metrics', fontsize=20)
    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)

    # Make the plots share the same x-axis
    axs[0].get_shared_x_axes().join(axs[0], axs[1])
    axs[0].set_xticklabels([])
    axs[1].set_xlabel('epoch', fontsize=20)

    # Add Y labels
    axs[1].set_ylabel('AP', fontsize=20) 

    # Adjust the spacing between the subplots
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
