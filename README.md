# ukbb-mri-sseg
UKBB MRI semantic segmentation for Abdominal Dixon and other modalities

This repository uses [git lfs](https://git-lfs.github.com/) for trained model files (inside `trained_models`).
Please use `git lfs pull` or `git lfs clone`.

## Worked Example with processed UKBB Data
We use an external published repository to process the raw UKBB data:
https://github.com/recoh/pipeline/

Please follow the installation instructions there to download and install.
After installing, activate the command line environment, and download process these datasets.
The following files will be created. `ESS3TE4SJ3DEP5C5` is a subject id. Please place the subject ID folder inside a `processed_nifti` folder for the purpose of this example. 
```
ESS3TE4SJ3DEP5C5/
├── nifti
│   ├── fat.nii.gz
│   ├── ideal_liver_magnitude.nii.gz
│   ├── ideal_liver_phase.nii.gz
│   ├── ip.nii.gz
│   ├── mask.nii.gz
│   ├── multiecho_pancreas_magnitude.nii.gz
│   ├── multiecho_pancreas_phase.nii.gz
│   ├── op.nii.gz
│   ├── t1_vibe_pancreas.nii.gz
│   ├── t1_vibe_pancreas_norm.nii.gz
│   └── water.nii.gz
```

### Install and create virtual environment
Please exit the CLI interface from `pipeline_private` repository, and activate the python virtual environment associated with this repository 
```
python3 -m venv env
source env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

We set up environment variables to be used in this worked example:
```
source env/bin/activate
SUBJECT_ID=ESS3TE4SJ3DEP5C5
echo $SUBJECT_ID > example_ids_list.txt
PROCESSED_NIFTI_DIR=processed_nifti
OUTPUT_FOLDER=/tmp/ukbb_mri_sseg_output
```

We download segmentation model weights. In the case of DIXON weights a further extraction step is needed.
```
git lfs pull
cat trained_models/knee_to_neck_dixon_split.tar.* | tar -xvf - -C trained_models/
```

The `tar` extraction command above might not work all systems. Alternatively,
```
cd trained_models
cat knee_to_neck_dixon_split.tar.* | tar -xvf -
cd ..
```


### Neck-to-knee DIXON Segmentation

This command performs inference on dixon data. See `/tmp/ukbb_mri_sseg_output` for output
```

DIXON_NIFTI_DIR=$PROCESSED_NIFTI_DIR/$SUBJECT_ID/nifti
RESTORE_STRING=trained_models/knee_to_neck_dixon/20200401-mpgp118-best_xe/model.ckpt-20000
python3 public_sseg/infer_knee_to_neck_dixon.py \
  --output_folder=$OUTPUT_FOLDER \
  --reference_header_nifti=$DIXON_NIFTI_DIR/mask.nii.gz \
  --save_what=prob \
  --threshold_method=fg \
  --fat=$DIXON_NIFTI_DIR/fat.nii.gz\
  --water=$DIXON_NIFTI_DIR/water.nii.gz\
  --inphase=$DIXON_NIFTI_DIR/ip.nii.gz\
  --outphase=$DIXON_NIFTI_DIR/op.nii.gz\
  --mask=$DIXON_NIFTI_DIR/mask.nii.gz\
  --restore_string=$RESTORE_STRING
```


### Pancreas 3D T1-weighted Segmentation
This command performs inference on dixon data. See `/tmp/ukbb_mri_sseg_output` for output
```
PANCREAS_MODEL_H5=trained_models/pancreas_t1w/20200104_shape-rep-a-dice2-ep50.h5
python3 public_sseg/infer_pancreas_t1w.py --action=infer \
    --output_folder=$OUTPUT_FOLDER \
    --IDS_FILE=example_ids_list.txt \
    --model_h5_path=$PANCREAS_MODEL_H5 \
    --input_nifti_subject_dir=$PROCESSED_NIFTI_DIR
```

### Liver IDEAL 2D Segmentation
This command performs inference on dixon data. See `/tmp/ukbb_mri_sseg_output` for output
```
LIVER_IDEAL_MODEL_H5=trained_models/liver_ideal/20191002_withphase_low_liver_2d_ideal.h5
python3 public_sseg/infer_liver_ideal_multiecho.py --action=infer \
    --output_folder=$OUTPUT_FOLDER \
    --data_modality=ideal_liver \
    --IDS_FILE=example_ids_list.txt \
    --model_h5_path=$LIVER_IDEAL_MODEL_H5 \
    --input_nifti_subject_dir=$PROCESSED_NIFTI_DIR
```

### Liver Multiecho 2D Segmentation
No worked example is available. The UKBB did release a reference resource file for this modality. If you already have the multiecho files downloaded and processed through the pipeline, and placed amongst the other nifti files inside `processed_nifti/ESS3TE4SJ3DEP5C5/nifti`, then command to segment the multiecho data would be:

```
LIVER_MULTIECHO_MODEL_H5=trained_models/liver_multiecho/20191002_withphase_low_liver_2d_multiecho.h5
python3 public_sseg/infer_liver_ideal_multiecho.py --action=infer \
    --output_folder=$OUTPUT_FOLDER \
    --data_modality=multiecho_liver \
    --IDS_FILE=example_ids_list.txt \
    --model_h5_path=$LIVER_IDEAL_MODEL_H5 \
    --input_nifti_subject_dir=$PROCESSED_NIFTI_DIR
```
