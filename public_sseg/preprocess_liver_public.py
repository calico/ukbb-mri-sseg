"""
Utilities to preprocess and visualize 2D MRI liver data: Multiecho and IDEAL
Useful for training segmentation and classification models
"""

import itertools
import os
import json
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.ndimage.morphology
import scipy.spatial.distance as distance
import skimage
import sklearn
import sklearn.metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from absl import logging


def histogram_equalize(img):
    img_cdf, bin_centers = skimage.exposure.cumulative_distribution(img, nbins=1024)
    return np.interp(img, bin_centers, img_cdf)


def load_image(image_path, equalize=True, return_nifti=False):
    if os.path.exists(image_path):
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()
        image_slice = image_data[:, :, 0, :]
        if equalize:
            image_slice = histogram_equalize(image_slice)
    else:
        logging.warn("{} does not exist".format(image_path))
        image_slice, image_nii = None, None
    if return_nifti:
        return image_slice, image_nii
    else:
        return image_slice


def load_mask(mask_path):
    """
    Load mask nifti file from mask_path. Data should be [H, W, D=1, C]
    Use dilation-fill-erode to remove holes / concavities from annotation artifacts.
    returns a [H, W] shaped mask
    Reduces away the channel dimension with a max.
    """
    if os.path.exists(mask_path):
        mask_nii = nib.load(mask_path)
        mask_data = np.max(mask_nii.get_fdata()[:, :, 0, :], axis=-1)
        mask_data = scipy.ndimage.binary_dilation(mask_data, iterations=2)
        mask_data = scipy.ndimage.morphology.binary_fill_holes(mask_data.astype(np.int)).astype(np.float64)
        mask_data = scipy.ndimage.morphology.binary_erosion(mask_data, iterations=2)
        mask_slice = np.stack([1 - mask_data[:, :, ],
                               mask_data[:, :, ]], axis=2)
    else:
        logging.warn("{} does not exist".format(mask_path))
        mask_slice = None
    return mask_slice


def preprocess_magnitude_phase(magnitude_raw, phase_raw, magnitude_mult,
                               phase_mult, name=None):
    if magnitude_raw is None:
        return None
    if phase_raw is None:
        return None
    magnitude = magnitude_mult * magnitude_raw
    mM = magnitude.max()
    if mM > 10.:
        logging.warn("({}) final magnitude has large max magnitude: {}".format(name, mM))
    if mM < .1:
        logging.warn("({}) final magnitude has small max magnitude: {}".format(name, mM))
    phase = phase_mult * phase_raw
    pM, pm = phase.max(), phase.min()
    if not np.isclose(pM - pm, np.pi * 2, rtol=1e-02, atol=1e-02):
        logging.warn("processed phase has support not close to 2pi: {}, {}".format(pm, pM))

    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)
    try:
        magnt_psin_pcos = np.concatenate([magnitude, phase_sin, phase_cos], axis=-1)
    except ValueError:
        logging.error("({}) failed to concat magnitude, phase_sin, phase_cos. {} {} {}".format(
            name, repr(magnitude.shape), repr(phase_sin.shape), repr(phase_cos.shape)))
        return None

    return magnt_psin_pcos


def plot_mask_image(image_slice, mask_slice):
    """
    visual diagnostics combining an image with its liver mask
    """
    if (image_slice is not None) and (mask_slice is not None):
        fig, axes = plt.subplots(1, 4)
        plt.axis('off')
        both = np.concatenate([image_slice, mask_slice], axis=2)

        axes[0].imshow(both[:, :, [3, 5, 7]], origin="lower")
        axes[1].imshow(both[:, :, [3, 11, 7]], origin="lower")
        axes[2].imshow(both[:, :, [0, 11, 9]], origin="lower")
        axes[3].imshow(both[:, :, [4, 11, 6]], origin="lower")

    return image_slice, mask_slice


def image_sanity_fail(image, shape, description):
    """
    Sanity check on images: training and testing; shape needs to match.
    description affects the logging, on failure.
    """
    if image is None:
        logging.error("{} : image is None".format(description))
        return True
    elif image.shape != shape:
        logging.error("{} : shape is {}, (expecting {})".format(
            description, repr(image.shape), repr(shape)))
        return True
    else:
        return False


def mask_sanity_fail(mask, shape, description):
    """
    Sanity check on training masks; shape needs to match.
    description affects the logging, on failure.
    """
    if mask is None:
        logging.warn("{} : mask is None".format(description))
        return True
    if mask.shape != shape:
        logging.warn("{} : shape is {}, (expecting {})".format(
            description, repr(mask.shape), repr(shape)))
        return True
    mm = mask[..., 1].mean()
    if mm > .5 or mm < .02:  # mostly should be .07 to .12
        logging.warn("{} : foreground mean {}".format(description, mm))
        return True


def np_rescale(_x, axis):
    M = np.max(_x, axis=axis, keepdims=True)
    m = np.min(_x, axis=axis, keepdims=True)
    d = M - m + 1e-5
    return (_x - m) / d


def rankdata_each(x_hwc):
    h, w, c = x_hwc.shape
    z_hwc = np.stack([np_rescale(rankdata(x_hwc[..., _].flatten()), axis=0)
                      for _ in range(c)], axis=1).reshape(list(x_hwc.shape))
    return z_hwc


def _pca_rank_scale_rgb_tiles(z_img, npc=7, triples=((0, 1, 2), (0, 3, 4), (0, 5, 6), None)):
    """
    z_img is of shape nhwc. nhw dimensions are flattened and the result is fit by a pca.
    triples specified
    """
    pca = PCA(n_components=npc)
    z = z_img.reshape([-1, z_img.shape[-1]])
    z = StandardScaler().fit_transform(z)
    pixel_pca_components = pca.fit_transform(z)
    pixel_pca_df = pd.DataFrame(data=pixel_pca_components, columns=['pc{}'.format(_) for _ in range(npc)])
    pixel_pca_arr = pixel_pca_df.to_numpy()

    tiles = []
    for t in triples:
        if t is None:  # top 3 principal components, without percentile-normalization
            arr = np_rescale(pixel_pca_arr[:, :3], axis=1).reshape(list(z_img.shape)[:-1] + [3, ])
        else:
            assert len(t) == 3
            arr = np.stack([np_rescale(rankdata(pixel_pca_arr[:, _]), axis=0)
                            for _ in t], axis=1).reshape(list(z_img.shape)[:-1] + [3, ])
        tiles.append(arr)
    return tiles


def visualize_head(r, c, vis_dict, image_name, indices=None, titles=None, suptitle=None):
    """
    Creates a figure of r by c subplots. The images are selected from vis_dict.
    The images to plot are keyed by a list of indices.
    titles are keyed in the same nmanner as vis_dict. Save the resulting figure as image_name.
    """
    if indices is None:
        indices = range(min(r * c, len(vis_dict)))
    plt.figure(figsize=(16, 16))

    for j, i in enumerate(indices):
        plt.subplot(r, c, j + 1)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i])
        plt.imshow(vis_dict[i])
        plt.axis('off')
    if suptitle is not None:
        plt.suptitle(suptitle)
    parent_dir, _ = os.path.split(image_name)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    plt.savefig(image_name)
    plt.close()


def get_srnp(gt, pred):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt, pred).ravel()
    return {'specificity': tn / (tn + fp),
            'recall': tp / (tp + fn),
            'negativepv': tn / (tn + fn),
            'precision': tp / (tp + fp)}


def evaluate_srnpdj(gt, pred, K=25):
    """
    Parameters:
        gt: np.array([N, ...])
            ground truth array of binary labels (integers)
        pred: np.array([, ...])
            same shape as ground truth
    Returns:
        srnpdj_N6: np.array(shape=[N, 6])
            for each paired item in (gt, pred), reteurn 5 evaluation numberse:
            specificity, recall, npv, ppv (prcision), dice-sim, jaccard.
        idx_low_metric_dict: dict(str: [int])
            dictionary of indices with lowest specificities, recall, npv, ppv, and dicesim
    """

    dicesim = []
    jaccard = []
    specificity = []
    recall = []
    negativepv = []
    precision = []

    for _y_true, _y_pred in zip(gt, pred):
        _y_true, _y_pred = _y_true.flatten(), _y_pred.flatten()
        srnp = get_srnp(_y_true, _y_pred)
        specificity.append(srnp['specificity'])
        recall.append(srnp['recall'])
        negativepv.append(srnp['negativepv'])
        precision.append(srnp['precision'])

        dicesim.append(1. - distance.dice(_y_true, _y_pred))
        jaccard.append(sklearn.metrics.jaccard_score(_y_true, _y_pred))

    specificity = np.array(specificity)
    recall = np.array(recall)
    negativepv = np.array(negativepv)
    precision = np.array(precision)
    dicesim = np.array(dicesim)
    jaccard = np.array(jaccard)
    logging.info("dicesim shape: {}".format(repr(dicesim.shape)))

    lowest_specificity = specificity.argsort()[:K]
    lowest_recall = recall.argsort()[:K]
    lowest_npv = negativepv.argsort()[:K]
    lowest_ppv = precision.argsort()[:K]
    lowest_dicesim = dicesim.argsort()[:K]
    lowest_jaccard = jaccard.argsort()[:K]

    srnpdj_N6 = np.stack([specificity, recall, negativepv, precision, dicesim, jaccard], axis=1)

    idx_low_metric_dict = {
        'specificity': lowest_specificity,
        'recall': lowest_recall,
        'npv': lowest_npv,
        'ppv': lowest_ppv,
        'dicesim': lowest_dicesim,
        'jaccard': lowest_jaccard,
    }
    return srnpdj_N6, idx_low_metric_dict


def visualize_predictions(x_img, gt, pred, names, prefix, img_dir, r=None, c=5, rot90=0):
    """
    Creates several figure of r by c subplots. The images are saved into img_dir. rot90 is an
    integer indicating the number of right angle ccw rotations to do. 4 means do nothing.
    prefix is used in the file names of images saved to disk.
    x_img is a bunch of multi channel raw images. We use rankorder and pca to visualize the
    pixels.

    gt and pred are ground truth and prediction masks. We visualize them overlayed on one of the input channels.
    We compute the ids of x_img with the lowest specificity, recall, npv, ppv, and dice.
    We plot the worst cases for each.
    Returns:
        srnpdj_N6: np.array(shape=[N, 6])
            for each paired item in (gt, pred), reteurn 5 evaluation numberse:
            specificity, recall, npv, ppv (prcision), dice-sim, jaccard.
    """

    if r is None:
        r = min(5, int(len(gt) / c))
    K = r * c

    srnpdj_N6, idx_low_metric_dict = evaluate_srnpdj(gt, pred, K=K)

    blue_cidx = 0  # pick any channel
    blue = np_rescale(x_img[:, :, :, blue_cidx:blue_cidx + 1], axis=None)
    red, green = gt, pred
    logging.info("rgb: {} {} {}".format(red.shape, green.shape, blue.shape))
    rgb = np.concatenate([red, green, blue], axis=3)

    vis_idx = set(itertools.chain(*idx_low_metric_dict.values()))
    vis_idx = vis_idx.union(range(r * c))
    visuals_dict = dict()  # compute only the images that score badly on some metric
    for i in vis_idx:
        tiles = _pca_rank_scale_rgb_tiles(x_img[i, :, :, :], triples=((0, 1, 2), (0, 0, 0), None))
        tiles_A = np.concatenate([tiles[0], tiles[1]], axis=0)
        tiles_B = np.concatenate([rgb[i], tiles[2]], axis=0)
        tiles_AB = np.concatenate([tiles_A, tiles_B], axis=1)
        tiles_AB = np.rot90(tiles_AB, k=rot90)
        visuals_dict[i] = tiles_AB

    visualize_head(r, c, visuals_dict, os.path.join(img_dir, 'sseg',
                                                    '{}_sseg.png'.format(prefix)),
                   indices=range(r * c), titles=names, suptitle='{}_{}'.format(prefix, 'samples'))
    for metric_name, low_idx in idx_low_metric_dict.items():
        visualize_head(r, c, visuals_dict, os.path.join(img_dir, 'low_{}'.format(metric_name),
                                                        '{}_low_{}.png'.format(prefix, metric_name)),
                       indices=low_idx, titles=names, suptitle='{}_low_{}'.format(prefix, metric_name))

    low_metric_ids = {k: [names[_] for _ in v] for k, v in idx_low_metric_dict.items()}
    with open(os.path.join(img_dir, '{}_low_metric_ids.json'.format(prefix)), 'w') as json_file:
        json.dump(low_metric_ids, json_file, indent=4)

    plt.figure(figsize=(16, 16))

    for i in range(4 * 5):
        tiles = _pca_rank_scale_rgb_tiles(x_img[i, :, :, :], triples=((0, 1, 2), (0, 3, 4), (0, 0, 0), None))
        tiles_A = np.concatenate([tiles[0], tiles[1]], axis=0)
        tiles_B = np.concatenate([tiles[2], tiles[3]], axis=0)
        tiles_AB = np.concatenate([tiles_A, tiles_B], axis=1)
        tiles_AB = np.rot90(tiles_AB, k=rot90)
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.title(names[i])
        plt.imshow(tiles_AB[:, :, :])

    plt.savefig(os.path.join(img_dir, '{}_visual.png'.format(prefix)))
    plt.close()
    return srnpdj_N6


def make_thresh_conservative(_x, _t, q=75):  # q = 75: keep 1 quarter of the mask.
    # takes in probabilities _x and float threshold _t. returns _t2, a more stringent threshold.
    _flat = _x.flatten()
    _idx = _flat >= _t
    _t2 = np.percentile(_flat[_idx], q)
    return _t2

# Use for scalar regression models


class ImageMaskNiftiReader(object):
    def __init__(self, magnitude_path_dict, phase_path_dict, mask_path_dict):
        self.magnitude_path_dict = magnitude_path_dict
        self.phase_path_dict = phase_path_dict
        self.mask_path_dict = mask_path_dict

    def get_combined_image_mask(self, subject_id):
        magnitude_path = self.magnitude_path_dict[subject_id]
        phase_path = self.phase_path_dict[subject_id]
        mask_path = self.mask_path_dict[subject_id]

        loaded_magnitude, magnitude_nifti = load_image(magnitude_path, equalize=False, return_nifti=True)
        loaded_phase = load_image(phase_path, equalize=False)
        loaded_magn_sin_cos = preprocess_magnitude_phase(loaded_magnitude, loaded_phase, magnitude_mult=1 / 200.,
                                                         phase_mult=np.pi / 4096., name=subject_id)
        loaded_mask = load_mask(mask_path)
        return combine_image_mask(loaded_magn_sin_cos, loaded_mask, method="concat")


def combine_image_mask(image, mask, method="concat"):
    if method == 'concat':
        return np.concatenate([image, mask], axis=-1)
    elif method == 'fg_only':
        raise NotImplementedError()
    elif method == 'bg_fg':
        bg, fg = np.split(mask, 2, axis=-1)
        bg_im = bg * image
        fg_im = fg * image
        return np.concatenate([bg_im, fg_im], axis=-1)
    else:
        raise ValueError("unknown combine_image_mask method: {}".format(method))


def yield_supervision(ids, im_mask_reader, float_df, isna_df=None, batch_size=4,
                      rand_rot_deg=0, rand_translate=0, skip_all_na=True,
                      shuffle=True, num_loops=None, skip_partial_batch=True,
                      HWC_check=(256, 232, 20), yield_format=None):
    if num_loops is None:
        loop_iter = iter(int, 1)  # infinite loop
    else:
        loop_iter = range(num_loops)

    for _loop in loop_iter:
        im_mask_list, float_list, isna_list, batch_ids = [], [], [], []
        idx = list(range(len(ids)))
        if shuffle:
            random.shuffle(idx)
        for _i in idx:
            curr_id = ids[_i]
            try:
                curr_float = float_df.loc[curr_id].values
                if isna_df is not None:
                    curr_isna = isna_df.loc[curr_id].values
                else:
                    curr_isna = ~np.isfinite(curr_float)
                if np.all(curr_isna) and skip_all_na:
                    # this entry does not have any valid supervision
                    # logging.info('skipping {} -- all supervision na'.format(curr_id))
                    continue
                else:
                    curr_float[curr_isna] = 0.

                im_mask = im_mask_reader.get_combined_image_mask(curr_id)
                if im_mask.shape != HWC_check:
                    logging.warn('invalid im_mask shape for id={} (is {}. expected {})-- skipping item and continuing in generator'.format(
                        curr_id, repr(im_mask.shape), repr(HWC_check)))
                    continue
                # simple data augmentation
                if rand_rot_deg:
                    deg = np.random.uniform(-rand_rot_deg, rand_rot_deg)
                    im_mask = scipy.ndimage.rotate(im_mask, deg, reshape=False)
                if rand_translate:
                    padded = np.pad(im_mask,
                                    ((rand_translate, rand_translate),
                                     (rand_translate, rand_translate), (0, 0)), mode='constant')
                    sh = np.random.randint(0, rand_translate * 2)
                    sw = np.random.randint(0, rand_translate * 2)
                    eh = sh + im_mask.shape[0]
                    ew = sw + im_mask.shape[1]
                    im_mask = padded[sh:eh, sw:ew, :]

            except (FileNotFoundError, ValueError, KeyError) as e:
                logging.warn(repr(e))
                logging.warn('unable to read data for id={} -- skipping item and continuing in generator'.format(curr_id))
                continue
            im_mask_list.append(im_mask)
            float_list.append(curr_float)
            isna_list.append(curr_isna)
            batch_ids.append(curr_id)
            if len(im_mask_list) == batch_size:
                try:
                    batch_im_mask = np.stack(im_mask_list, axis=0)
                    batch_float = np.stack(float_list, axis=0)
                    batch_isna = np.stack(isna_list, axis=0)
                    split_float_targets = [_[:, 0] for _ in
                                           np.split(batch_float, batch_float.shape[1], axis=1)]
                    split_isna_targets = [_[:, 0] for _ in
                                          np.split(1. - batch_isna, batch_float.shape[1], axis=1)]
                except ValueError as e:
                    logging.warn(repr(e))
                    logging.warn('unable to stack or split for ids={} -- skipping batch and continuing in generator'.format(repr(batch_ids)))
                    continue

                inputs = [batch_im_mask, batch_float, batch_isna]
                outputs = split_float_targets
                sample_weights = split_isna_targets
                if yield_format == 'inputs_ids':
                    yield inputs, batch_ids
                else:
                    yield inputs, outputs, sample_weights
                im_mask_list, float_list, isna_list, batch_ids = [], [], [], []
        # after looping through
        if len(im_mask_list) > 0:
            if not skip_partial_batch:
                try:
                    batch_im_mask = np.stack(im_mask_list, axis=0)
                    batch_float = np.stack(float_list, axis=0)
                    batch_isna = np.stack(isna_list, axis=0)
                    split_float_targets = [_[:, 0] for _ in
                                           np.split(batch_float, batch_float.shape[1], axis=1)]
                    split_isna_targets = [_[:, 0] for _ in
                                          np.split(1. - batch_isna, batch_float.shape[1], axis=1)]
                except ValueError as e:
                    logging.warn(repr(e))
                    logging.warn('unable to stack or split for ids={} -- skipping final partial batch and continuing in generator'.format(repr(batch_ids)))
                    continue

                inputs = [batch_im_mask, batch_float, batch_isna]
                outputs = split_float_targets
                sample_weights = split_isna_targets
                if yield_format == 'inputs_ids':
                    yield inputs, batch_ids
                else:
                    yield inputs, outputs, sample_weights
                im_mask_list, float_list, isna_list, batch_ids = [], [], [], []
