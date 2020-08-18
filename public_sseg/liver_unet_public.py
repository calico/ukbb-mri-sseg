"""
Standard U-NET Model of 2D liver segmentation.

Implementation adapted from https://github.com/zhixuhao/unet/blob/master/model.py
"""
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K  # https://github.com/keras-team/keras/issues/4609

def make_keras_unet_model(pretrained_weights=None, input_size=(256, 256, 1), num_classes=2, lr=1e-5,
                          activation='relu', optimizer_decay=1e-6, gpus=1):
    regularizer_opts = dict(
        kernel_regularizer=keras.regularizers.l2(1e-4),
        #    bias_regularizer=keras.regularizers.l2(1e-4),
        #    activity_regularizer=keras.regularizers.l2(1e-4),
    )

    def truncate_a1a2_cat3(x):
        x0, xtrunc = x
        _, H, W, _ = x0.get_shape().as_list()
        x1 = xtrunc[:, :H, :W, :]
        x0x1 = K.concatenate([x0, x1], axis=-1)
        return x0x1

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(inputs)
    conv1 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(pool1)
    conv2 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(pool2)
    conv3 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(pool3)
    conv4 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv4)
    # drop4 = Dropout(0.5)(conv4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(bn4)

    conv5 = Conv2D(1024, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(pool4)
    conv5 = Conv2D(1024, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv5)
    # drop5 = Dropout(0.5)(conv5)
    bn5 = BatchNormalization()(conv5)

    up6 = Conv2D(512, 2, activation=activation, padding='same', kernel_initializer='he_normal',
                 **regularizer_opts)(UpSampling2D(size=(2, 2))(bn5))
    # merge6 = concatenate([bn4, up6], axis=3)
    merge6 = Lambda(truncate_a1a2_cat3)([bn4, up6])
    conv6 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(merge6)
    conv6 = Conv2D(512, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv6)

    up7 = Conv2D(256, 2, activation=activation, padding='same', kernel_initializer='he_normal',
                 **regularizer_opts)(UpSampling2D(size=(2, 2))(conv6))

    #merge7 = concatenate([conv3, up7], axis=3)
    merge7 = Lambda(truncate_a1a2_cat3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(merge7)
    conv7 = Conv2D(256, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv7)

    up8 = Conv2D(128, 2, activation=activation, padding='same', kernel_initializer='he_normal',
                 **regularizer_opts)(UpSampling2D(size=(2, 2))(conv7))
    # merge8 = concatenate([conv2, up8], axis=3)
    merge8 = Lambda(truncate_a1a2_cat3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(merge8)
    conv8 = Conv2D(128, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv8)

    up9 = Conv2D(64, 2, activation=activation, padding='same', kernel_initializer='he_normal',
                 **regularizer_opts)(UpSampling2D(size=(2, 2))(conv8))
    # merge9 = concatenate([conv1, up9], axis=3)
    merge9 = Lambda(truncate_a1a2_cat3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(merge9)
    conv9 = Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_normal', **regularizer_opts)(conv9)
    probs = Conv2D(
        num_classes,
        3,
        activation='softmax',
        padding='same',
        kernel_initializer='he_normal',
        **regularizer_opts,
        name='probs')(conv9)

    model = Model(inputs=inputs, output=probs)
    if gpus > 1:
        model = keras.utils.multi_gpu_model(
            model,
            gpus=gpus,
            cpu_merge=True,
            cpu_relocation=False
        )
    print(probs)

    model.compile(optimizer=Adam(lr=lr, decay=optimizer_decay), loss=loss_sseg,  # loss_sseg, #K.categorical_crossentropy, # keras.losses.categorical_crossentropy
                  metrics=['accuracy', keras.metrics.categorical_crossentropy, dice2_loss])  # , keras.metrics.Recall(class_id=1)])

    model.summary()
    print("optimizer_decay {}".format(optimizer_decay))

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def dice2_coef(y_true, y_pred, axis=(1, 2), smooth=1e-5):
    intersection = tf.reduce_sum((y_true * y_pred), axis=axis)
    d2 = (2. * intersection + smooth) / (tf.reduce_sum(K.square(y_true), axis=axis) + tf.reduce_sum(K.square(y_pred), axis=axis) + smooth)
    print(y_true, y_pred, d2)
    return d2


def dice2_loss(labels, probs, axis=(1, 2)):
    return 1 - dice2_coef(labels, probs, axis)


def loss_sseg(labels, probs, fg_idx_start=0):
    xe = K.categorical_crossentropy(labels, probs)
    xe = tf.reduce_mean(xe, axis=(1, 2), name='xe')  # reduce away hw
    d2 = tf.identity(dice2_loss(labels[:, :, :, fg_idx_start:], probs[:, :, :, fg_idx_start:], axis=(1, 2)), name='d2')
    d2 = tf.reduce_mean(d2, axis=1, name='d2')     # reduce away c
    xe_d2 = xe + d2
    print(xe, d2, xe_d2)
    return xe_d2
