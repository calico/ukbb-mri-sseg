"""
action = infer
- Loads a saved model h5 file from previous training
- Reads a list of subject ides, and navigates a folder of nifti files
- Infers segmentation; copies nifti header over to the output. The output has shape [H, W, 1, 1]

The visualize and train actions rely on ground truth annotationos being available.

action = visualize
- Loads a saved model h5 file from previous training
- Creates a panels of png visualizations
- Focus on instances with poor segmentation performance; visualize mask ablation

action = train
- Trains a unet semantic segmentation model for 2d images of shape (160, 160, 10) or (256, 232, 6)
- Saves a saved model h5 file
- Creates a bunch of png visualizations
- Input Images have shapes:
    Magnitudes: [H, W, 1, C] -Postives upto ~500
    Phase: [H, W, 1, C]. Ranges from -4096 to 4096 representing -pi and +pi
    Mask: [H, W, 1, C]. Binary values in {0 (bg), 1 (fg)}. Exactly one of the channels is populated.

Uses Tensorflow 1.4 Keras.
"""
import json
import keras
import liver_unet_public as unet
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import skimage.filters as filters

from absl import flags, logging, app
from keras.preprocessing.image import ImageDataGenerator
from preprocess_liver_public import load_image, load_mask, \
    preprocess_magnitude_phase, image_sanity_fail, mask_sanity_fail, \
    make_thresh_conservative, visualize_predictions, rankdata_each

FLAGS = flags.FLAGS


def main(unused):
    del unused
    logging.set_verbosity(logging.INFO)
    annotations_subject_dir = FLAGS.annotations_subject_dir # where annotation files are stored
    IDS_FILE = FLAGS.IDS_FILE
    IMAGE_DIR = FLAGS.input_nifti_subject_dir
    model_fullpath = FLAGS.model_h5_path      # e.g. trained_models/liver_ideal/20191002_withphase_low_liver_2d_ideal.h5
    model_subdir, model_filename = os.path.split(model_fullpath)
    model_name = model_filename[:-3] if model_filename.endswith('.h5') else model_filename

    individual_ids_file = os.path.join(IDS_FILE)
    with open(individual_ids_file) as fp:
        individual_ids = [i.rstrip() for i in fp.readlines()]

    if FLAGS.data_modality == 'multiecho_liver':
        IMAGE_NAME = 'multiecho_liver_magnitude.nii.gz'
        PHASE_NAME = 'multiecho_liver_phase.nii.gz'
        MASK_NAME = 'multiecho_liver.nii.gz'
        image_shape3 = (160, 160, 30) if FLAGS.use_phase else (160, 160, 10)
    elif FLAGS.data_modality == 'ideal_liver':
        IMAGE_NAME = 'ideal_liver_magnitude.nii.gz'
        PHASE_NAME = 'ideal_liver_phase.nii.gz'
        MASK_NAME = 'ideal_liver.nii.gz'
        image_shape3 = (256, 232, 18) if FLAGS.use_phase else (256, 232, 6)
    else:
        raise ValueError("Unknown data modality: {}".format(FLAGS.data_modality))

    IMAGE_SUBDIR = 'nifti'
    MASK_SUBDIR = 'annotations'

    def get_image_path(individual_id):
        return os.path.join(IMAGE_DIR, individual_id, IMAGE_SUBDIR, IMAGE_NAME)

    def get_phase_path(individual_id):
        return os.path.join(IMAGE_DIR, individual_id, IMAGE_SUBDIR, PHASE_NAME)

    def get_mask_path(individual_id):
        return os.path.join(annotations_subject_dir, individual_id, MASK_SUBDIR, MASK_NAME)

    image_path_list = [get_image_path(_id) for _id in individual_ids]
    mask_path_list = [get_mask_path(_id) for _id in individual_ids]
    H, W, C = image_shape3
    loaded_mask = np.zeros([len(individual_ids), H, W, 2]) if FLAGS.action == 'infer' else [load_mask(_) for _ in mask_path_list]
    if FLAGS.use_phase:
        phase_path_list = [get_phase_path(_id) for _id in individual_ids]
        loaded_magnitude, magnitude_niftis = zip(
            *[load_image(_, equalize=False, return_nifti=True) for _ in image_path_list])
        loaded_phase = [load_image(_, equalize=False) for _ in phase_path_list]
        loaded_magn_sin_cos = [preprocess_magnitude_phase(_r, _p, magnitude_mult=1 / 200.,
                                                          phase_mult=np.pi / 4096., name=_id) for _r, _p, _id in zip(
            loaded_magnitude, loaded_phase, individual_ids)]
        load_data_list = zip(loaded_magn_sin_cos, loaded_mask)
    else:
        loaded_magnitude, magnitude_niftis = zip(
            *[load_image(_, equalize=True, return_nifti=True) for _ in image_path_list])
        load_data_list = zip(loaded_magnitude, loaded_mask)
    sanity_data_list = []
    sanity_individual_ids = []
    sanity_fail_ids = []
    sanity_nifti_list = []
    if 'rankdata' in FLAGS.preprocess:
        logging.info("rankdata preprocessing")
    for _i, _ip, _mp, (image, mask), nifti in zip(
            individual_ids, image_path_list, mask_path_list, load_data_list, magnitude_niftis):
        failed = False
        if image_sanity_fail(image, shape=image_shape3, description="{} {}".format(_i, _ip)):
            failed = True
        elif FLAGS.action != 'infer' and mask_sanity_fail(mask, shape=(H, W, 2),
                                                          description="{} {}".format(_i, _mp)):
            logging.warn("mask sanity fail: {}. {}".format(_i, _mp))
            failed = True

        if failed:
            sanity_fail_ids.append(_i)
        else:
            image_preproc = image
            if 'rankdata' in FLAGS.preprocess:
                image_preproc = rankdata_each(x_hwc=image_preproc)
            sanity_data_list.append((image, mask, image_preproc))
            sanity_individual_ids.append(_i)
            sanity_nifti_list.append(nifti)

    data_list = sanity_data_list

    logging.info("len(data_list)={}".format(len(data_list)))
    frac_train = FLAGS.frac_train
    num_train = int(len(data_list) * frac_train)

    train_idxs = range(len(data_list))[:num_train]
    test_idxs = range(len(data_list))[num_train:]
    train_data_list = [data_list[_] for _ in train_idxs]
    test_data_list = [data_list[_] for _ in test_idxs]
    train_individual_ids = [sanity_individual_ids[_] for _ in train_idxs]
    test_individual_ids = [sanity_individual_ids[_] for _ in test_idxs]

    x_all, y_all, z_all = list(zip(*data_list))
    x_all = np.stack(x_all, axis=0)
    y_all = np.stack(y_all, axis=0).astype(np.int)
    z_all = np.stack(z_all, axis=0)

    x_train = x_all[:num_train]
    x_test = x_all[num_train:]
    y_train = y_all[:num_train]
    y_test = y_all[num_train:]
    z_train = z_all[:num_train]
    z_test = z_all[num_train:]

    if not os.path.exists(model_subdir):
        os.makedirs(model_subdir)
    
    if FLAGS.action == 'train':
        # we create two instances with the same arguments
        if FLAGS.augment == 'full':
            data_gen_args = dict(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=90,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='nearest',
                                 zoom_range=0.2,
                                 )
        elif FLAGS.augment == 'flipreduced':
            data_gen_args = dict(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=10,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=.05,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='nearest',
                                 zoom_range=0.05)
        elif FLAGS.augment == 'reduced':
            data_gen_args = dict(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=10,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=.05,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 fill_mode='nearest',
                                 zoom_range=0.05)
        elif FLAGS.augment == 'low':
            data_gen_args = dict(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=5,
                                 width_shift_range=0.02,
                                 height_shift_range=0.02,
                                 shear_range=.02,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 fill_mode='nearest',
                                 zoom_range=0.02)
        else:
            raise ValueError()
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        image_datagen.fit(z_train, augment=True, seed=seed)
        mask_datagen.fit(y_train, augment=True, seed=seed)

        per_gpu_batch_size = 32
        global_batch_size = per_gpu_batch_size * FLAGS.num_gpus
        print("per_gpu_batch_size: {}".format(per_gpu_batch_size))
        print("global_batch_size: {}".format(global_batch_size))

        image_generator = image_datagen.flow(z_train, seed=seed, batch_size=global_batch_size,)
        mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=global_batch_size,)
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False,)
        session_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_mem_f
        session_config.gpu_options.allow_growth = True

        sess = tf.compat.v1.Session(config=session_config)
        set_session(sess)
        input_size = z_all.shape[1:]
        logging.info('input_size {}'.format(input_size))

        model = unet.make_keras_unet_model(input_size=input_size, lr=1e-4, num_classes=2,
                                           activation=FLAGS.activation, optimizer_decay=0.,
                                           gpus=FLAGS.num_gpus)

        logging.info("keras.backend.get_value(model.optimizer.lr) {}".format(keras.backend.get_value(model.optimizer.lr)))
        logging.info("x max/mean/min: {} {} {}".format(x_train.max(), x_train.mean(), x_train.min()))
        logging.info("z max/mean/min: {} {} {}".format(z_train.max(), z_train.mean(), z_train.min()))
        logging.info("y mean by channel: {} {}".format(y_train[:, :, :, 0].mean(), y_train[:, :, :, 1].mean()))
        model.reset_states()

        ep = FLAGS.ep
        spe = 20  # // FLAGS.num_gpus # steps per epoch
        lr_sched_1gpu = np.array([1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]) if FLAGS.use_phase \
            else np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-7, 1e-7])
        lr_sched = lr_sched_1gpu  # * FLAGS.num_gpus
        for lr in lr_sched:
            keras.backend.set_value(model.optimizer.lr, lr)
            logging.info("lr = {}".format(lr))
            model.fit_generator(
                train_generator,
                steps_per_epoch=spe,
                validation_data=(z_test, y_test),
                use_multiprocessing=True,
                workers=4,
                epochs=ep)
        logging.info('saving to {}'.format(model_fullpath))
        if not os.path.exists(model_subdir):
            os.makedirs(model_subdir)
        model.save(model_fullpath)
    elif FLAGS.action in ['visualize', 'infer']:
        from liver_unet_public import loss_sseg, dice2_loss
        from keras.models import load_model
        logging.info('loading from {}'.format(model_fullpath))
        model = load_model(model_fullpath, custom_objects={'loss_sseg': loss_sseg, 'dice2_loss': dice2_loss})
    else:
        raise ValueError("Unknown action {}".format(FLAGS.action))
    if FLAGS.action == 'infer':
        all_pred_prob = model.predict(z_all)
        logging.info("id, thresh, otsu_mask_px, after_erosion_size")
        out_subdir = {"multiecho_liver": "multiecho_liver_seg",
                      "ideal_liver": "ideal_liver_seg"}[FLAGS.data_modality]
        seg_out_folder = os.path.join(FLAGS.output_folder, out_subdir)
        if not os.path.exists(seg_out_folder):
            os.makedirs(seg_out_folder)

        def _save_nifti(nparr, fullpath, affine=None, header=None):
            nib_img = nib.Nifti1Image(nparr, affine, header)
            nib.save(nib_img, fullpath)
        for _pred_prob, _id, nifti in zip(all_pred_prob, sanity_individual_ids, sanity_nifti_list):
            _thresh = filters.threshold_otsu(_pred_prob[:, :, 1:2]).item()
            _pred = (_pred_prob[:, :, 1] > _thresh).astype(int)
            _flat_prob = _pred_prob[:, :, 1].flatten()
            _thresh2 = make_thresh_conservative(_flat_prob, _thresh, q=FLAGS.infer_q)
            _pred2 = (_pred_prob[:, :, 1] > _thresh2).astype(int)
            header = nifti.header.copy()
            header.set_data_dtype(np.int8)
            # go from [H, W] to [H, W, 1, 1]
            nparr = _pred2.reshape([H, W, 1, 1])
            nifti_fullpath = os.path.join(seg_out_folder, "{}.nii.gz".format(_id))
            logging.info('{} thresholds: {}, {}. sizes: {}, {}.'.format(
                _id, _thresh, _thresh2, _pred.sum(), _pred2.sum()))
            _save_nifti(nparr=nparr, fullpath=nifti_fullpath, affine=nifti.affine, header=header)
    else:  # visualize or train
        train_pred_prob = model.predict(z_train)
        test_pred_prob = model.predict(z_test)

        train_thresh = np.array([filters.threshold_otsu(p) for p in train_pred_prob[:, :, :, 1:2]]).reshape([-1, 1, 1, 1])
        test_thresh = np.array([filters.threshold_otsu(p) for p in test_pred_prob[:, :, :, 1:2]]).reshape([-1, 1, 1, 1])

        train_pred = (train_pred_prob[:, :, :, 1:2] >= train_thresh).astype(int)
        test_pred = (test_pred_prob[:, :, :, 1:2] >= test_thresh).astype(int)

        train_srnpdj = visualize_predictions(x_train, y_train[:, :, :, 1:2], pred=train_pred, names=[
                                             str(_) for _ in train_individual_ids], prefix='train', img_dir=model_subdir,
                                             rot90=FLAGS.visual_rot90)
        val_srnpdj = visualize_predictions(x_test, y_test[:, :, :, 1:2], pred=test_pred, names=[
                                           str(_) for _ in test_individual_ids], prefix='val', img_dir=model_subdir,
                                           rot90=FLAGS.visual_rot90)

        # qs = np.array(range(5, 80, 5)).astype(np.int)
        qs = [25]  # final paper
        train_consthr = np.array([make_thresh_conservative(_x, _t.item(), q=qs) for _x, _t in zip(
            train_pred_prob[..., 1], train_thresh.flatten())]).reshape([-1, 1, 1, len(qs)])
        test_consthr = np.array([make_thresh_conservative(_x, _t.item(), q=qs) for _x, _t in zip(
            test_pred_prob[..., 1], test_thresh.flatten())]).reshape([-1, 1, 1, len(qs)])
        print("train_consthr.shape", train_consthr.shape)

        overall_evaluation = {}
        srnpdj_str = ['specificity', 'recall', 'npv', 'precision', 'dice-sim', 'jaccard']
        print("sanity fails: {}".format(repr(sanity_fail_ids)))
        print("srnpdj: specificity, recall, negative predictive value, precision (ppv), dice-sim")

        np.set_printoptions(precision=3)
        print("mean train srnpdj:")
        print(train_srnpdj.mean(axis=0))
        print("mean val srnpdj:")
        print(val_srnpdj.mean(axis=0))
        overall_evaluation[('train_{}'.format(0))] = {k: v for k, v in zip(srnpdj_str, train_srnpdj.mean(axis=0))}
        overall_evaluation[('val_{}'.format(0))] = {k: v for k, v in zip(srnpdj_str, val_srnpdj.mean(axis=0))}

        for i, q in enumerate(qs):
            print("conservative q={:02d}".format(q))
            train_pred2 = (train_pred_prob[:, :, :, 1:2] > train_consthr[:, :, :, i:i + 1]).astype(int)
            test_pred2 = (test_pred_prob[:, :, :, 1:2] > test_consthr[:, :, :, i:i + 1]).astype(int)
            train_conservative_srnpdj = visualize_predictions(x_train, y_train[:, :, :, 1:2], pred=train_pred2, names=[
                str(_) for _ in train_individual_ids], prefix='train_cons{:02d}'.format(q), img_dir=model_subdir,
                rot90=FLAGS.visual_rot90)
            val_conservative_srnpdj = visualize_predictions(x_test, y_test[:, :, :, 1:2], pred=test_pred2, names=[
                str(_) for _ in test_individual_ids], prefix='val_convs{:02d}'.format(q), img_dir=model_subdir,
                rot90=FLAGS.visual_rot90)

            print("mean train_conservative srnpdj:")
            print(train_conservative_srnpdj.mean(axis=0))
            print("mean val_conservative srnpdj:")
            print(val_conservative_srnpdj.mean(axis=0))
            overall_evaluation[('train_{}'.format(q))] = {k: v for k, v in zip(srnpdj_str, train_conservative_srnpdj.mean(axis=0))}
            overall_evaluation[('val_{}'.format(q))] = {k: v for k, v in zip(srnpdj_str, val_conservative_srnpdj.mean(axis=0))}

        with open(os.path.join(model_subdir, 'overall_evaluation.json'), 'w') as json_file:
            json.dump(overall_evaluation, json_file, indent=4)


if __name__ == "__main__":
    # Model specific paramenters
    flags.DEFINE_string("IDS_FILE", "example_ids_list.txt",
                        "Text file containing ids to train and/or validate on, one per line")
    flags.DEFINE_string(
        "output_folder",
        "/tmp/ukbb_mri_sseg_output/",
        "When doing inference, where to dump the output niftii")
    flags.DEFINE_string("annotations_subject_dir", "processed_nifti/",
                        "parent path from which to read annotation nifti files")
    flags.DEFINE_string("input_nifti_subject_dir", "processed_nifti/",
                        "parent path from which to read images and masks")

    flags.DEFINE_string("model_h5_path", "trained_models/liver_ideal/20191002_withphase_low_liver_2d_ideal.h5",
                        "Model h5 file to load")
    flags.DEFINE_string("augment", "full", "augmentation string: full or reduced")
    flags.DEFINE_string(
        "action",
        "infer",
        "(train) model from data; (visualize) previously trained model on training data; (infer) segmentation on data without annotations.")
    flags.DEFINE_boolean("use_phase", True, "Use phase information")
    flags.DEFINE_integer("num_gpus", 1, "number of GPUs to use for training")
    flags.DEFINE_integer("visual_rot90", 1, "rot90 multiplier to apply in visualization ")
    flags.DEFINE_integer("ep", 5, "number of epochs")

    flags.DEFINE_float("frac_train", 0.9,  # 1024,
                       "Fraction of data to use for training. ignored when action is inference")
    flags.DEFINE_float("gpu_mem_f", 2.5,
                       """https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
    If greater than 1.0, uses CUDA unified memory to potentially oversubscribe
    the amount of memory available on the GPU device by using host memory as a
    swap space. (Pascal+ Only)
    """)
    flags.DEFINE_string("preprocess", "", "preprocessing steps. substrings checked: rankdata")
    flags.DEFINE_string("activation", "relu", "activation function")
    flags.DEFINE_string("data_modality", "ideal_liver", "data modality: multiecho_liver or ideal_liver")
    flags.DEFINE_float("infer_q", 25,  # 1024,
                       "Inference only. Remove this percent of each segmentation mask, based on confidence.")

    logging.set_verbosity(logging.WARN)
    app.run(main)
