"""
Inference for 3D pancreas segmentation
"""

import h5py
import json
import nibabel as nib
import numpy as np
import os
import pathlib
import scipy.spatial.distance as distance
import sklearn.metrics
import tensorflow as tf

from absl import flags, logging, app
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import optimizers as KO

FLAGS = flags.FLAGS

N_TRANSLATIONS = 10


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype('float32') / np.percentile(img, 99)
    img = np.minimum(img, 2.)
    return img


def load_nifti_as_1xyz1(subject_dir):
    nib_path = os.path.join(subject_dir, 'nifti', 't1_vibe_pancreas_norm.nii.gz')
    loaded = nib.load(nib_path)
    data_1xyz1 = loaded.get_data()[np.newaxis, :, :, :, np.newaxis]
    return data_1xyz1, loaded


# # truncate x1 to the shape of x0, then return the sum
def truncate_a1a2a3_add(x0_x1):
    x0, x1 = x0_x1
    _, X, Y, Z, _ = tf.unstack(tf.shape(x0))
    x1 = x1[:, :X, :Y, :Z, :]
    return x0 + x1


def truncate_a1a2a3_x1(x0_x1):  # truncate x1 to the shape of x0, then return x1
    x0, x1 = x0_x1
    _, X, Y, Z, _ = tf.unstack(tf.shape(x0))
    x1 = x1[:, :X, :Y, :Z, :]
    return x1


def load_pancreas_data(data_h5_path: str, n_translations: int = N_TRANSLATIONS, frac_train: float = .7):
    with h5py.File(data_h5_path, "r") as hf:
        data = hf['data'][:]
        labels = hf['labels'][:]

    n_obs = int(np.round(data.shape[0] / n_translations * frac_train)) * n_translations
    data_train = data[:n_obs, :, :, :, :]
    labels_train = labels[:n_obs, :, :, :, :]
    data_validation = data[n_obs::n_translations, :, :, :, :]  # pick 1 translations amongst many
    labels_validation = labels[n_obs::n_translations, :, :, :, :]
    logging.info("n_obs = {}".format(n_obs))
    logging.info("data_train shape = {}".format(repr(data_train.shape)))
    logging.info("data_validation shape = {}".format(repr(data_validation.shape)))
    logging.info("labels_train shape = {}".format(repr(labels_train.shape)))
    logging.info("labels_validation shape = {}".format(repr(labels_validation.shape)))
    return data_train, labels_train, data_validation, labels_validation


def load_model_h5(model_h5_path):
    logging.info('loading from {}'.format(model_h5_path))
    keras_model = tf.keras.models.load_model(model_h5_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'dice2_coefficient': dice2_coefficient,
        'binary_crossentropy': binary_crossentropy,
        'truncate_a1a2a3_add': truncate_a1a2a3_add,
        'truncate_a1a2a3_x1': truncate_a1a2a3_x1})
    keras_model.layers.pop(0)  # modify input res_
    keras_model.summary()
    return keras_model


def get_keras_otsu_model():
    import tf_otsu
    otsu_in = KL.Input([None])
    otsu_out = tf_otsu.tf_otsu(otsu_in)
    keras_otsu_model = tf.keras.models.Model(inputs=[otsu_in], outputs=[otsu_out],)
    keras_otsu_model.summary()
    return keras_otsu_model


def _dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def _dice2_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true ** 2, axis=[1, 2, 3]) + K.sum(y_pred ** 2, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - _dice_coef(y_true, y_pred)


def dice2_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - _dice2_coef(y_true, y_pred)


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _xe = K.binary_crossentropy(y_true, y_pred)
    _xe = K.mean(_xe, axis=[1, 2, 3])
    return _xe


def pancreas_3d_unet(input_size, l2_reg=1e-4, use_layer_normalization=None):
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    inputs = KL.Input(input_size)
    if FLAGS.downsample == 'maxpool':
        downsampled = KL.MaxPooling3D(pool_size=(2, 2, 1), padding='same')(inputs)
    elif FLAGS.downsample == 'conv':
        downsampled = KL.Conv3D(FLAGS.C_mid, (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(inputs)
    elif FLAGS.downsample == 'cc':
        conv0 = KL.Conv3D(FLAGS.C_mid, (5, 5, 5), activation='relu', padding='same')(inputs)
        downsampled = KL.Conv3D(FLAGS.C_mid, (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(conv0)
    elif FLAGS.downsample == 'c2p':
        conv0 = KL.Conv3D(FLAGS.C_mid, (5, 5, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(inputs)
        conv0 = KL.Conv3D(FLAGS.C_mid, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv0)
        downsampled = KL.MaxPooling3D(pool_size=(2, 2, 1), padding='same')(conv0)
    else:
        raise ValueError("downsample method: {}".format(FLAGS.downsample))
    logging.info("use_layer_normalization: {}".format(use_layer_normalization))

    def make_keras_norm():
        if use_layer_normalization:
            return KL.LayerNormalization()
        else:
            return KL.BatchNormalization()

    conv1 = KL.Conv3D(FLAGS.C_mid * 2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(downsampled)
    conv1 = make_keras_norm()(conv1)
    conv1 = KL.Conv3D(FLAGS.C_mid * 2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv1)
    conv1 = make_keras_norm()(conv1)
    pool1 = KL.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv2 = KL.Conv3D(FLAGS.C_mid * 4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(pool1)
    copy2 = conv2
    conv2 = make_keras_norm()(conv2)
    conv2 = KL.Conv3D(FLAGS.C_mid * 4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv2)
    conv2 = KL.add([copy2, conv2])
    conv2 = make_keras_norm()(conv2)
    pool2 = KL.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)

    conv3 = KL.Conv3D(FLAGS.C_mid * 8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(pool2)
    copy3 = conv3
    conv3 = make_keras_norm()(conv3)
    conv3 = KL.Conv3D(FLAGS.C_mid * 8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv3)
    conv3 = KL.add([copy3, conv3])
    conv3 = make_keras_norm()(conv3)
    pool3 = KL.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

    conv4 = KL.Conv3D(FLAGS.C_mid * 16, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(pool3)
    copy4 = conv4
    conv4 = make_keras_norm()(conv4)
    conv4 = KL.Conv3D(FLAGS.C_mid * 16, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv4)
    conv4 = KL.add([copy4, conv4])
    conv4 = make_keras_norm()(conv4)
    pool4 = KL.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv4)

    conv5 = KL.Conv3D(FLAGS.C_mid * 16, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(pool4)
    copy5 = conv5
    conv5 = make_keras_norm()(conv5)
    conv5 = KL.Conv3D(FLAGS.C_mid * 16, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv5)
    conv5 = KL.add([copy5, conv5])
    conv5 = make_keras_norm()(conv5)

    up6 = KL.Lambda(truncate_a1a2a3_add)([conv4, KL.UpSampling3D(size=(2, 2, 2))(conv5)])
    conv6 = KL.Conv3D(FLAGS.C_mid * 8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(up6)
    conv6 = make_keras_norm()(conv6)
    conv6 = KL.Conv3D(FLAGS.C_mid * 8, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv6)
    conv6 = make_keras_norm()(conv6)

    up7 = KL.Lambda(truncate_a1a2a3_add)([conv3, KL.UpSampling3D(size=(2, 2, 2))(conv6)])
    conv7 = KL.Conv3D(FLAGS.C_mid * 4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(up7)
    conv7 = make_keras_norm()(conv7)
    conv7 = KL.Conv3D(FLAGS.C_mid * 4, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv7)
    conv7 = make_keras_norm()(conv7)

    up8 = KL.Lambda(truncate_a1a2a3_add)([conv2, KL.UpSampling3D(size=(2, 2, 2))(conv7)])
    conv8 = KL.Conv3D(FLAGS.C_mid * 2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(up8)
    conv8 = make_keras_norm()(conv8)
    conv8 = KL.Conv3D(FLAGS.C_mid * 2, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv8)
    conv8 = make_keras_norm()(conv8)

    up9 = KL.Lambda(truncate_a1a2a3_add)([conv1, KL.UpSampling3D(size=(2, 2, 2))(conv8)])
    conv9 = KL.Conv3D(FLAGS.C_mid, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(up9)
    conv9 = make_keras_norm()(conv9)
    conv9 = KL.Conv3D(FLAGS.C_mid, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(conv9)
    conv9 = make_keras_norm()(conv9)

    finalup = KL.Lambda(truncate_a1a2a3_x1)([inputs, KL.UpSampling3D(size=(2, 2, 1))(conv9)])
    conv11 = KL.Conv3D(FLAGS.C_mid, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=kernel_regularizer)(finalup)
    conv11 = KL.Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_regularizer=kernel_regularizer)(conv11)
    return inputs, conv11


def main(unused):
    del unused
    logging.set_verbosity(logging.INFO)
    if FLAGS.action == 'preprocess':
        raise NotImplementedError()
    elif FLAGS.action == 'train':
        raise NotImplementedError()
    elif FLAGS.action == 'evaluate':
        raise NotImplementedError()
    elif FLAGS.action == 'infer':
        keras_model = load_model_h5(FLAGS.model_h5_path)
        keras_otsu_model = get_keras_otsu_model()
        with open(FLAGS.IDS_FILE) as f:
            all_subject_ids = [_.strip() for _ in f]
            all_subject_ids = [_ for _ in all_subject_ids if len(_) > 0]

        end_idx = len(all_subject_ids) if FLAGS.how_many is None \
            else FLAGS.start_idx + FLAGS.how_many
        subject_ids = all_subject_ids[FLAGS.start_idx:end_idx]
        logging.info('perfoming inference on {} of {} subject ids'.format(len(subject_ids), len(all_subject_ids)))

        def _save_nifti(nparr, fullpath, affine=None, header=None):
            nib_img = nib.Nifti1Image(nparr, affine, header)
            nib.save(nib_img, fullpath)

        for _id in subject_ids:
            try:
                subject_nifti_dir = os.path.join(FLAGS.input_nifti_subject_dir, _id)
                logging.info('loading subject {} from {}'.format(_id, subject_nifti_dir))
                curr_data, nifti = load_nifti_as_1xyz1(subject_nifti_dir)
                curr_data = normalize(curr_data)

            except Exception as e:
                logging.error("subject id {} experienced error: {}".format(_id, repr(e)))
                continue
            model_outs = keras_model.predict(curr_data, batch_size=1, verbose=1)
            sseg_probs = model_outs[0]  # several copies are returned. take one.
            # import skimage.filters as filters
            # _thresh = filters.threshold_otsu(sseg_probs).item()
            _thresh, = keras_otsu_model.predict(sseg_probs.reshape([1, -1]))
            sseg_binary = (sseg_probs > _thresh).squeeze().astype(np.int8)
            logging.info("otsu thresh = {}. fg pixels: {}.".format(_thresh, sseg_binary.sum()))

            seg_out_folder = os.path.join(FLAGS.output_folder, "pancreas_t1w_sseg")
            if not os.path.exists(seg_out_folder):
                os.makedirs(seg_out_folder)
            header = nifti.header.copy()
            header.set_data_dtype(np.int8)
            nifti_fullpath = os.path.join(seg_out_folder, "{}.nii.gz".format(_id))
            _save_nifti(nparr=sseg_binary, fullpath=nifti_fullpath, affine=nifti.affine, header=header)
    else:
        raise ValueError("{}: Unknown action".format(FLAGS.action))


if __name__ == "__main__":
    flags.DEFINE_string(
        "action",
        "infer",
        "perform preprocessing on data; train model from data; visualize previously trained model on training data; infer segmentation on data without annotations.")
    flags.DEFINE_float("gpu_mem_f", 2.5,
                       """https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
    If greater than 1.0, uses CUDA unified memory to potentially oversubscribe
    the amount of memory available on the GPU device by using host memory as a
    swap space. (Pascal+ Only)
    """)

    # model settings
    flags.DEFINE_string("loss_type", "dice1", "dice1, dice2, xe")
    flags.DEFINE_string("downsample", "c2p", "maxpool, conv, cc, c2p")
    flags.DEFINE_integer("C_mid", 16, "base channels multiple")

    # train settings
    flags.DEFINE_string("models_dir", "trained_models/", "Parent dir under which to load models")

    # inference settings
    flags.DEFINE_string("IDS_FILE", "example_ids_list.txt",
                        "Text file containing ids to do inference on, one per line")
    flags.DEFINE_string("input_nifti_subject_dir", "processed_nifti/",
                        "Directory containing: individual_id/nifti/t1_vibe_pancreas_norm.nii.gz")
    flags.DEFINE_string("model_h5_path", None, "Full path a specific keras model h5 file to use in inference")
    flags.DEFINE_string(
        "output_folder",
        "/tmp/ukbb_mri_sseg_output/",
        "When doing inference, where to dump the output nifti files")
    flags.DEFINE_integer('how_many', None, "How many subject IDs to process. Processes all if specified")
    flags.DEFINE_integer('start_idx', 0, "Starting index of subject ID.")
    app.run(main)
