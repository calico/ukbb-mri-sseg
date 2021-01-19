"""
Inference loop

Infer directly from five nifti files: fat, water, in-phase, out-of-phase, mask.
This is the ordering of channels used by the neural network.

The first fouor channels: fat, water, in-phase, out-of-phase have an underlying linear relationship
in complex spsace to each other. The input is expected to have been subsequently normalized, and
so the relationship is lost. Each channel is positive and has standard deviation 1,  with respect to the
UKBB dataset. We are robust to somewhat negative numbers.
The means we used were respectively: []

The last channel is a mask channel. It takes value 1 if the pixels belongs to a person, and 0 if background.

The input nifti files should have shape [W=224, H=174, D=370, C=5] This is the input shape/ordering expected
by this script. The underlying neural network used a different spatial order, and this wrapper script
is responsible for that conversion.
(For reference, the underlying neural network expected the order BDHWC where B is batch size)

This network requires that input data be oriented correctly. We describe the direction specified
by going along each axis in the numpy arrays, from a low-integer index to a high integer index.
W=244. Each slice is saggital. From low index to high: subject's right to subject's left.
H=174. Each slice is coronal. From low index to high: subject's posterior to subject's dorsal
D=370. Each slice is axial.  From low index to high: subject's knee to neck.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import pickle
import nibabel as nib
import tensorflow as tf
from absl import logging
from absl import flags
from absl import app
from odictliteral import odict

import random
import datetime
import skimage.filters as filters
import scipy.special as spsp
import skimage.measure as measure

import unet_trunk_public


FLAGS = flags.FLAGS

# this one assumes no batch dimension, unlike in tf_graph.py. TODO: refactor.


def tf_otsu(x,):
    x = tf.reshape(x, [-1])
    z_fwd = tf.sort(x, axis=-1, direction='ASCENDING')  # numerical precision considerations
    var_fwd = tf_cumulative_var(z_fwd)

    z_rev = tf.reverse(z_fwd, axis=[-1])
    var_rev = tf_cumulative_var(z_rev)

    p_fwd = tf.cast(tf.range(1, 1 + tf.shape(z_fwd)[0]), dtype=tf.float32) / tf.cast(tf.shape(x), tf.float32)
    q_fwd = 1. - p_fwd
    var_both = p_fwd * var_fwd + q_fwd * tf.reverse(var_rev, axis=[-1])
    z_am = tf.argmin(var_both)
    return z_fwd[z_am]


def tf_cumulative_var(z):
    z2 = z ** 2
    cu_sz = tf.cumsum(z)
    cu_sz2 = tf.cumsum(z2)
    n = tf.cast(tf.range(tf.shape(z)[-1]), dtype=tf.float32) + 1
    cu_mz = cu_sz / n
    cu_mz2 = cu_sz2 / n
    cu_var = cu_mz2 - (cu_mz ** 2)
    return cu_var


def bg_based_thresholding(np_list_logit_masks_WHDC, _srun_tf_otsu=None):
    np_list_prob_masks_WHDC = [spsp.softmax(_, axis=-1) for _ in np_list_logit_masks_WHDC]
    if _srun_tf_otsu is None:
        list_bg_logit_otsu_thresh = [filters.threshold_otsu(_[..., 0].flatten()) for _ in np_list_logit_masks_WHDC]
        list_bg_prob_otsu_thresh = [filters.threshold_otsu(_[..., 0].flatten()) for _ in np_list_prob_masks_WHDC]
    else:
        list_bg_logit_otsu_thresh = [_srun_tf_otsu(_[..., 0].reshape([1, -1])) for _ in np_list_logit_masks_WHDC]
        list_bg_prob_otsu_thresh = [_srun_tf_otsu(_[..., 0].reshape([1, -1])) for _ in np_list_prob_masks_WHDC]

    logging.info("raw bg prob otsu thresh: {}".format(repr(list_bg_prob_otsu_thresh)))
    list_bg_prob_otsu_thresh = [max(_, FLAGS.min_prob_thresh) for _ in list_bg_prob_otsu_thresh]
    logging.info("final bg prob otsu thresh: {}".format(repr(list_bg_prob_otsu_thresh)))

    logging.info("bg_logit_otsu_thresh {}".format(repr(list_bg_logit_otsu_thresh)))

    np_list_fg_logit_masks_WHDC = [_.copy() for _ in np_list_logit_masks_WHDC]
    np_list_fg_prob_masks_WHDC = [_.copy() for _ in np_list_prob_masks_WHDC]
    np_list_otsu_logit_masks_WHDC = []
    np_list_otsu_logit_argmax_masks_WHD = []
    np_list_otsu_prob_masks_WHDC = []
    np_list_otsu_prob_argmax_masks_WHD = []
    for _fg_logit, _raw_logit, _fg_prob, _raw_prob, _logit_thresh, _prob_thresh \
            in zip(np_list_fg_logit_masks_WHDC, np_list_logit_masks_WHDC,
                   np_list_fg_prob_masks_WHDC, np_list_prob_masks_WHDC,
                   list_bg_logit_otsu_thresh, list_bg_prob_otsu_thresh):
        _neginf = -1000  # set bg to -1000 in the foreground only version of logits
        _fg_logit[..., 0] = _neginf
        _fg_prob[..., 0] = 0  # a copy of _raw_prob, but with class 0 (bg) zero-ed out.

        _otsu_logit = np.where(_raw_logit[..., :1] > _logit_thresh, _raw_logit, _fg_logit)
        _otsu_prob = np.where(_raw_prob[..., :1] > _prob_thresh, _raw_prob, _fg_prob)  # doesn't add to 1
        np_list_otsu_logit_masks_WHDC.append(_otsu_logit)
        np_list_otsu_logit_argmax_masks_WHD.append(np.argmax(_otsu_logit, axis=-1).astype(np.int8))
        np_list_otsu_prob_masks_WHDC.append(_otsu_prob)
        np_list_otsu_prob_argmax_masks_WHD.append(np.argmax(_otsu_prob, axis=-1).astype(np.int8))
    return dict(
        np_list_otsu_logit_masks_WHDC=np_list_otsu_logit_masks_WHDC,
        np_list_otsu_logit_argmax_masks_WHD=np_list_otsu_logit_argmax_masks_WHD,
        np_list_otsu_prob_masks_WHDC=np_list_otsu_prob_masks_WHDC,
        np_list_otsu_prob_argmax_masks_WHD=np_list_otsu_prob_argmax_masks_WHD,
        np_list_prob_masks_WHDC=np_list_prob_masks_WHDC)


def fg_based_thresholding(np_list_logit_masks_WHDC, _srun_tf_otsu):
    np_list_prob_masks_WHDC = [spsp.softmax(_, axis=-1) for _ in np_list_logit_masks_WHDC]

    list2_fg_logit_otsu_thresh = []  # list of list [[... for c in _classes] for o in organs]
    list2_fg_prob_otsu_thresh = []

    def _list2_otsu(_np_list_values_WHDC, _min_thresh=None):
        list2_values_otsu_thresh = []
        for values in _np_list_values_WHDC:
            C = values.shape[-1]
            values_otsu_thresh = []
            for c in range(C):
                values_otsu_thresh.append(_srun_tf_otsu(values[..., c].reshape([1, -1])))
                if _min_thresh is not None:
                    values_otsu_thresh = [max(_, _min_thresh) for _ in values_otsu_thresh]
            list2_values_otsu_thresh.append(np.array(values_otsu_thresh))
        return list2_values_otsu_thresh
    list2_fg_logit_otsu_thresh = _list2_otsu(np_list_logit_masks_WHDC, _min_thresh=None)
    list2_fg_prob_otsu_thresh = _list2_otsu(np_list_prob_masks_WHDC, _min_thresh=FLAGS.min_prob_thresh)
    logging.info("final prob otsu thresh: {}".format(repr(list2_fg_prob_otsu_thresh)))
    logging.info("list2_fg_prob_otsu_thresh {}".format(repr(list2_fg_prob_otsu_thresh)))

    def _apply_thresh_WHDC(x_WHDC, thresh):
        # apply threshold with special bg treatment (c=0)
        binary_WHDC = x_WHDC > thresh
        binary_WHDC[..., 0] = np.logical_and(binary_WHDC[..., 0],
                                             ~np.amax(binary_WHDC[..., 1:], axis=-1))
        return binary_WHDC

    np_list_otsu_logit_masks_WHDC = [_apply_thresh_WHDC(_x, _t) for _x, _t in zip(
        np_list_logit_masks_WHDC, list2_fg_logit_otsu_thresh)]
    np_list_otsu_prob_masks_WHDC = [_apply_thresh_WHDC(_x, _t) for _x, _t in zip(
        np_list_prob_masks_WHDC, list2_fg_prob_otsu_thresh)]

    def _argmax_WHDC(x_WHDC, is_fg_WHD):
        # argmax but avoid giving bg=0 label, depnding on otsu of all the fg channels
        _am = np.argmax(x_WHDC, axis=-1).astype(np.int8)
        return np.select([is_fg_WHD], [_am], default=0)

    np_list_otsu_logit_argmax_masks_WHD = [
        _argmax_WHDC(_x, is_fg_WHD=~_bin[..., 0])
        for _x, _bin in zip(np_list_logit_masks_WHDC, np_list_otsu_logit_masks_WHDC)]
    np_list_otsu_prob_argmax_masks_WHD = [
        _argmax_WHDC(_x, is_fg_WHD=~_bin[..., 0])
        for _x, _bin in zip(np_list_prob_masks_WHDC, np_list_otsu_prob_masks_WHDC)]
    return dict(
        np_list_otsu_logit_masks_WHDC=np_list_otsu_logit_masks_WHDC,
        np_list_otsu_logit_argmax_masks_WHD=np_list_otsu_logit_argmax_masks_WHD,
        np_list_otsu_prob_masks_WHDC=np_list_otsu_prob_masks_WHDC,
        np_list_otsu_prob_argmax_masks_WHD=np_list_otsu_prob_argmax_masks_WHD,
        np_list_prob_masks_WHDC=np_list_prob_masks_WHDC)


def keep_largest_connected_components(x, fg_frac=.98):
    if fg_frac > 1.:
        raise ValueError('fg_frac cannot be greater than 1. (is {})'.format(fg_frac))
    components, N = measure.label(x, background=0, return_num=True, connectivity=3)
    # logging.info('connected component {}'.format(components))
    sizes = np.bincount(components.flatten())
    logging.info('connected component sizes {}'.format(sizes))
    fg_total_size = np.sum(sizes[1:])
    fg_idx_sorted = sorted(list(range(1, N + 1)),
                           key=lambda idx: sizes[idx], reverse=True)
    cumu_px = 0.
    fg_indices = []
    for idx in fg_idx_sorted:
        if cumu_px >= fg_frac * fg_total_size:
            break
        fg_indices.append(idx)
        cumu_px += sizes[idx]
    logging.info("connected component fg_indices: {}".format(fg_indices))
    return np.isin(components, fg_indices)

def manual_strided_tile(psize, fullsize, stride):
    """
    Suppose f is a 1-D fully convolutional function, whose inputs and outputs have the same shape on spatial dimensions.
    Supppose that we do not have enough memory to call f on some particular input, because its spatial dimension is too large.

    We want to seemlessly tile a convolutional call f, unrolled over iterations, so as for the output to be identical to a single calll -- if the stride is sufficiently small.
    If the stride is too high, then the tiling will not be seemless. If the stride is too low, then there will be unnecessary computations

    Parameters:
        given total size, patch size, and stride.

    returns:
        i_starts: start indices for the input, across each iteration
        a_starts: start indices for the accumulator, across each iteration
        o_starts: start indices for the output of the fun
    """
    o_start = (psize - stride) // 2

    i_starts = list(range(0, fullsize - psize + 1, stride))
    if i_starts[-1] + psize < fullsize:
        i_starts.append(fullsize - psize)
    i_starts = np.array(i_starts, dtype=np.int)
    a_starts = i_starts + o_start
    a_starts[0] = 0

    a_starts = i_starts + (psize // 2) - (stride // 2)
    a_starts[0] = 0

    a_ends = np.concatenate([a_starts[1:], np.array([fullsize])])
    o_starts = a_starts - i_starts
    o_ends = a_ends - i_starts
    return dict(i_starts=i_starts, i_ends=i_starts + psize, a_starts=a_starts, a_ends=a_ends, o_starts=o_starts, o_ends=o_ends)


def main(_ignore):
    del _ignore
    logging.set_verbosity(logging.INFO)

    ORGAN_NAMES = sorted(
        ["abdominal_cavity", "body_cavity", "kidney_left",
         "kidney_right", "liver", "lungs", "spleen"])
    ORGAN_PARTID_NAMES = odict([(_, odict[0: "bg", 1: _]) for _ in ORGAN_NAMES])
    list_num_semantic_classes = [len(_v.values()) for _k, _v in ORGAN_PARTID_NAMES.items()]

    params = {
        'batch_size': 1,
        'normlayer': FLAGS.normlayer,
        "unet_blocks": FLAGS.unet_blocks,
        "C_mid": FLAGS.C_mid,
        "C_max": FLAGS.C_max,
        "unet_dtype": tf.float32,

        "ORGAN_PARTID_NAMES": ORGAN_PARTID_NAMES,
        "list_num_semantic_classes": list_num_semantic_classes,
    }

    list_num_semantic_classes = params["list_num_semantic_classes"]

    arch_args_dict = {
        'normalization': params['normlayer'],
        'kernel_regularizer': None,
    }
    D, H, W, C = (370, 174, 224, 5)

    restore_latest_dir = FLAGS.restore_latest_dir
    #out_dir = FLAGS.out_dir
    if FLAGS.output_folder is None:
        hashval = random.getrandbits(24)
        out_dir = "/tmp/{}{:X}".format(datetime.datetime.now().strftime("%m%d-%H%M-"), hashval)
    else:
        out_dir = os.path.join(FLAGS.output_folder, 'knee_to_neck_dixon_seg')
    logging.info('out_dir: {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.info("out_dir = {}".format(out_dir))

    image_NDHWC = tf.placeholder(dtype=tf.float32, shape=[1, None, H, W, C])  # we tile on D
    image_NDHWC = tf.cast(image_NDHWC, params.get('unet_dtype', tf.float32))
    unet_trunk_dict = unet_trunk_public.mri_unet_sseg_model_NDHWC(
        image_NDHWC=image_NDHWC,
        list_num_semantic_classes=list_num_semantic_classes,
        version='v1',
        C_mid=params.get('C_mid', 20),
        C_max=params.get('C_max', 4096),
        blocks=params.get('unet_blocks', None),
        arch_args_dict=arch_args_dict)

    restore_string = FLAGS.restore_string
    if restore_string is None:
        restore_string = tf.train.latest_checkpoint(restore_latest_dir)
    assert restore_string is not None
    logging.info("Restoring from restore_string: {}".format(restore_string))

    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem_f
    # if not FLAGS.f16:  # bug / errors are associated with xla on float16
    #     session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=session_config)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, restore_string)
    logging.info("{}".format(repr(unet_trunk_dict.keys())))

    tf_otsu_input = tf.placeholder(dtype=tf.float32, shape=[1, None])
    tf_otsu_thresh = tf_otsu(tf_otsu_input)

    def _srun_tf_otsu(_x, _sess=sess):
        return _sess.run(tf_otsu_thresh, feed_dict={tf_otsu_input: _x})

    if FLAGS.reference_header_nifti is None or \
            FLAGS.reference_header_nifti == "" or \
            FLAGS.reference_header_nifti == "None":
        logging.warn("reference_header_nifti is set to None. May result in unuseable nifti files")
        int8_header, float32_header = None, None
        ref_affine = np.eye(4)
    else:
        logging.info("reference_header_nifti file {}".format(FLAGS.reference_header_nifti))
        nib_ref_nifti = nib.load(FLAGS.reference_header_nifti)
        int8_header, float32_header = nib_ref_nifti.header.copy(), nib_ref_nifti.header.copy()
        int8_header.set_data_dtype(np.int8)
        float32_header.set_data_dtype(np.float32)
        ref_affine = nib_ref_nifti.affine
        # OK to not set_data_shape -- use reference shape
    P = 160
    stride = {160: 32, 192: 64, 224: 96}[P]
    stride_dict = manual_strided_tile(psize=P, fullsize=D, stride=stride)
    logging.info("P = {}".format(P))
    logging.info("stride_dict = {}".format(repr(stride_dict)))

    # fat, water, in, out, mask

    def read_and_transpose(nifti_path, einsum_str):
        x = nib.load(nifti_path).get_fdata()
        logging.info('read_and_transpose raw shape {}'.format(repr(x.shape)))
        return np.einsum(einsum_str, x)

    input_einsum_str = FLAGS.input_einsum_str
    output_einsum_str = FLAGS.output_einsum_str
    logging.info("input_einsum_str: {}".format(input_einsum_str))
    logging.info("output_einsum_str: {}".format(output_einsum_str))
    mri_pixels_list = [read_and_transpose(nifti_path, input_einsum_str) for nifti_path in
                       [FLAGS.fat, FLAGS.water, FLAGS.inphase, FLAGS.outphase, FLAGS.mask]]

    mri_pixels_1DHWC = np.expand_dims(np.stack(mri_pixels_list, axis=-1), axis=0)
    logging.info('before division max: {}'.format(repr(mri_pixels_1DHWC.max(axis=(0, 1, 2, 3)))))
    logging.info('before division min: {}'.format(repr(mri_pixels_1DHWC.min(axis=(0, 1, 2, 3)))))
    logging.info('before division std: {}'.format(repr(mri_pixels_1DHWC.std(axis=(0, 1, 2, 3)))))
    PIXEL_DIVISORS = np.array([176.0532375, 111.87631875, 197.93829375, 171.8048875, 1.])
    div_pixels_1DHWC = mri_pixels_1DHWC / PIXEL_DIVISORS
    logging.info('after division max: {}'.format(repr(div_pixels_1DHWC.max(axis=(0, 1, 2, 3)))))
    logging.info('after division min: {}'.format(repr(div_pixels_1DHWC.min(axis=(0, 1, 2, 3)))))
    logging.info('after division std: {}'.format(repr(div_pixels_1DHWC.std(axis=(0, 1, 2, 3)))))

    # split up the input into overlapping tiles -- then accumulate
    np_list_logit_masks_NDHWK = [np.zeros(shape=[1, D] + _.get_shape().as_list()[2:])
                                 for _ in unet_trunk_dict['list_logit_masks_NDHWK']]
    for _is, _ie, _as, _ae, _os, _oe in zip(*[stride_dict[k]
                                              for k in ['i_starts', 'i_ends', 'a_starts', 'a_ends', 'o_starts', 'o_ends']]):
        feed_dict = {image_NDHWC: div_pixels_1DHWC[:, _is:_ie, :, :, :]}
        curr_NDHWK = sess.run(unet_trunk_dict['list_logit_masks_NDHWK'], feed_dict=feed_dict)
        for _accumulator, _curr_out in zip(np_list_logit_masks_NDHWK, curr_NDHWK):
            _accumulator[0, _as:_ae, :, :, :] = _curr_out[0, _os:_oe, :, :, :]
        logging.info('patch session call on D=({}:{})'.format(_is, _ie))
    # np_list_logit_masks_DHWK is a list over organs, and K indexes the "classes" of each organ (bg/fg)
    np_list_logit_masks_DHWK = [_[0, ...] for _ in np_list_logit_masks_NDHWK]  # batch size = 1
    ### np_list_logit_masks_WHDC = [DHWC_to_WHDC(_) for _ in np_list_logit_masks_DHWK]
    np_list_logit_masks_WHDK = [np.einsum(output_einsum_str, _) for _ in np_list_logit_masks_DHWK]
    np_list_argmax_masks_WHD = [np.argmax(_, axis=-1).astype(np.int8) for _ in np_list_logit_masks_WHDK]

    if FLAGS.threshold_method == 'bg':
        # use heuristic to get overall bg prob
        # run otsu on that, then decode each fg with more heuristics
        thr_dict = bg_based_thresholding(np_list_logit_masks_WHDK, _srun_tf_otsu=_srun_tf_otsu)
    elif FLAGS.threshold_method == 'fg':  # otsu each fg class separately
        thr_dict = fg_based_thresholding(np_list_logit_masks_WHDK, _srun_tf_otsu=_srun_tf_otsu)
    else:
        raise ValueError("Unknown FLAGS.threshold_method {}".format(FLAGS.threshold_method))
    np_list_otsu_logit_argmax_masks_WHD = thr_dict['np_list_otsu_logit_argmax_masks_WHD']
    np_list_otsu_prob_argmax_masks_WHD = thr_dict['np_list_otsu_prob_argmax_masks_WHD']
    np_list_prob_masks_WHDC = thr_dict['np_list_prob_masks_WHDC']

    logging.info("len(np_list_logit_masks_NDHWK) = {}".format(len(np_list_logit_masks_NDHWK)))
    for i, v in enumerate(np_list_logit_masks_NDHWK):
        logging.info('{} {} {}'.format(i, v.shape, repr(v.dtype)))

    if "rawpkl" in FLAGS.save_what:
        _fname = os.path.join(out_dir, 'list_logit_masks_DHWK.pkl')
        logging.info("Saving rawpkl with logit, argmax/prob and argmaxes to {}".format(_fname))
        with open(_fname, 'wb+') as f:
            pickle.dump({"list_logit_masks_DHWK": np_list_logit_masks_DHWK,
                         "list_argmax_masks_WHD": np_list_argmax_masks_WHD,
                         "otsu_logit_argmax_masks_WHD": np_list_otsu_logit_argmax_masks_WHD,
                         "otsu_prob_argmax_masks_WHD": np_list_otsu_prob_argmax_masks_WHD, }, file=f)
    else:
        logging.info("Not saving rawpkl with logit, argmax/prob argmaxes")

    organ_names = list(ORGAN_PARTID_NAMES.keys())
    for i, organ in enumerate(organ_names):

        def _save_nifti(nparr, organ, fullpath, affine=None, header=None):
            logging.info("Saving {} to {}".format(organ, fullpath))
            if header is None:
                nib_img = nib.Nifti1Image(nparr, affine)
            else:
                nib_img = nib.Nifti1Image(nparr, affine, header)
            nib.save(nib_img, fullpath)

        if "logit" in FLAGS.save_what:
            nifti_fullpath = os.path.join(out_dir, "logit_{}.nii.gz".format(organ))
            _save_nifti(nparr=np_list_logit_masks_WHDC[i].astype(np.float32), organ=organ,
                        fullpath=nifti_fullpath, affine=ref_affine, header=float32_header)
        if "prob" in FLAGS.save_what:
            nifti_fullpath = os.path.join(out_dir, "prob_{}.nii.gz".format(organ))
            _save_nifti(nparr=np_list_prob_masks_WHDC[i].astype(np.float32), organ=organ,
                        fullpath=nifti_fullpath, affine=ref_affine, header=float32_header)

        if "miscamax" in FLAGS.save_what:
            nifti_fullpath = os.path.join(out_dir, "argmax_{}.nii.gz".format(organ))
            _save_nifti(nparr=np_list_argmax_masks_WHD[i], organ=organ,
                        fullpath=nifti_fullpath, affine=ref_affine, header=int8_header)

            nifti_fullpath = os.path.join(out_dir, "otsu_logit_argmax_{}.nii.gz".format(organ))
            _save_nifti(nparr=np_list_otsu_logit_argmax_masks_WHD[i], organ=organ,
                        fullpath=nifti_fullpath, affine=ref_affine, header=int8_header)

        # perform connected component post processing
        nifti_fullpath = os.path.join(out_dir, "otsu_prob_argmax_{}.nii.gz".format(organ))
        _cc_otsu_prob_argmax_masks_WHD = keep_largest_connected_components(np_list_otsu_prob_argmax_masks_WHD[i])
        _save_nifti(nparr=_cc_otsu_prob_argmax_masks_WHD, organ=organ,
                    fullpath=nifti_fullpath, affine=ref_affine, header=int8_header)


if __name__ == "__main__":
    flags.DEFINE_integer("unet_blocks", 5, "Blocks in u-net")
    flags.DEFINE_integer("C_mid", 72, "Number of channels in the outer sandwich layer of the U-net")
    flags.DEFINE_integer("C_max", 1152, "Number of channels in the inner sandwich layer of the U-net")
    flags.DEFINE_string("normlayer", "batchnorm", "normalization layers inside the main network: batchnorm, grounorm, identity")
    flags.DEFINE_string("output_folder", None, "Directory to which logit pkl files and logit nifti files will be saved")
    flags.DEFINE_string(
        "restore_latest_dir",
        None,
        "if restore_string is None, then use restore_latest_dir to generate a check point string")
    flags.DEFINE_string(
        "restore_string",
        None,
        "restore string. If left None, auto-generate by tf.train.latest_checkpoint(restore_latest_dir) ")
    flags.DEFINE_string(
        "input_einsum_str", "whd->dhw",
        "einsum string used to transpose each of the input numpy arrays after being read from nifti files")
    flags.DEFINE_string(
        "output_einsum_str", "dhwk->whdk",
        "einsum string used to transpose each of the model output arays before saving to disk.")

    flags.DEFINE_float("min_prob_thresh", 0.01, "min_prob_thresh: minimal probability threshold required to consider as foreground")

    flags.DEFINE_string("reference_header_nifti",
                        None,
                        "reference header nifti file from which header is to be copied")
    flags.DEFINE_string("save_what", "logit",
                        "Various argmaxes are always saved. Save additional nifti files if "
                        "the following substrings are specified in this string argument. "
                        "logit: logit values in (-inf, inf). "
                        "prob: prob values in (-inf, inf). ")
    flags.DEFINE_string("threshold_method", "fg",
                        "fg (otsu-threshold each foreground channel individually), bg (otsu-threshold bg only, then argmax on fg channels on complement)")
    flags.DEFINE_string(
        "fat",
        None,
        "path to the fat channel signals (expecting [W=224, H=174, D=370] shape). values should be positive with, approximately mean {} std {}".format(
            None,
            None))
    flags.DEFINE_string(
        "water",
        None,
        "path to the fat channel signals (expecting [W=224, H=174, D=370] shape). values should be positive with, approximately mean {} std {}".format(
            None,
            None))
    flags.DEFINE_string(
        "inphase",
        None,
        "path to the fat channel signals (expecting [W=224, H=174, D=370] shape). values should be positive with, approximately mean {} std {}".format(
            None,
            None))
    flags.DEFINE_string(
        "outphase",
        None,
        "path to the fat channel signals (expecting [W=224, H=174, D=370] shape). values should be positive with, approximately mean {} std {}".format(
            None,
            None))
    flags.DEFINE_string("mask", None, "path to the fat channel signals (expecting [W=224, H=174, D=370] shape). values should be 0 or 1.")
    flags.DEFINE_float("gpu_mem_f", 0.95,
                       """https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
    If greater than 1.0, uses CUDA unified memory to potentially oversubscribe
    the amount of memory available on the GPU device by using host memory as a
    swap space. (Pascal Only)
    """)
    logging.set_verbosity(logging.INFO)
    app.run(main)
