"""
U-net Based on Calico Tensorflow port of Allen Cell Network
Intended for use on UKBB Dixon MRI


Notes:
conv3_br: conv3, batchnorm, relu
"""
import tensorflow as tf
import numpy as np
from absl import logging
import copy


def mri_unet_sseg_model_NDHWC(
        image_NDHWC,
        variables_dict=None,
        is_training=False,
        list_num_semantic_classes=(1, ),
        version='v1',
        C_in=5,
        C_mid=20,
        C_max=4096,
        blocks=None,
        bn_momentum=0.99,
        arch_args_dict=None):
    """
    x: input tensor
    variables_dict: dict
        A dictionary into which variables and tensors are saved as the graph is being defined.
        For debugging / inspection purposes only
    training: bool
        whether we are in training mode
    """
    if blocks is None:
        blocks = 5
    if arch_args_dict is None:
        arch_args_dict = dict()

    if variables_dict is None:
        variables_dict = dict()
    trunk_NDHWC = allen_sub_u_net(image_NDHWC, variables_dict, C_in, L=C_mid, C_max=C_max, blocks=blocks,
                                  training=is_training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)
    list_logit_masks_NDHWK = []

    final_sseg_arch_args_dict = copy.copy(arch_args_dict)
    final_sseg_arch_args_dict["normalization"] = 'identity'

    for i, nsc in enumerate(list_num_semantic_classes):
        key = 'sseg_logits_{:02d}_{:02d}'.format(i, nsc)
        variables_dict[key] = dict()
        with tf.variable_scope(key):
            curr_sseg = conv3_br(trunk_NDHWC, C=C_mid, L=nsc,
                                 variables_dict=variables_dict[key],
                                 training=is_training, do_relu=False,
                                 arch_args_dict=final_sseg_arch_args_dict)
        list_logit_masks_NDHWK.append(curr_sseg)
    ret_dict = {
        'trunk_NDHWC': trunk_NDHWC,
        'variables_dict': variables_dict,
        'training': is_training,
        'bn_momentum': bn_momentum,
        'C_in': C_in,
        'C_mid': C_mid,
        'blocks': blocks,
        'version': version,
        'list_logit_masks_NDHWK': list_logit_masks_NDHWK,
    }
    return ret_dict


def set_dict_no_overwrite(d, k, v):
    """
    d: dict
    k: key
    v: value
    """
    assert k not in d
    d[k] = v


def allen_sub_u_net(x, variables_dict, C, C_mult=2, L=None, C_max=4096, down_stride=2,
                    blocks=4, training=True, bn_momentum=0.99, arch_args_dict=None):
    """
    Subnet portion of a u-net copied from the pytorch system. Calls itself recursively
    C_mult is hardcoded to 2 in recursive calls
    x: input tensor
    variables_dict: dict
        A dictionary into which variables and tensors are saved as the graph is being defined.
        For debugging purposes
    C: channels
    C_mult: int
        channel count multiplier in recursive call. If L is not specified, then L defaults to C_mult * C
    L: int
        number of output (latent) channels inside this particular call and not passed down in recursion.
        Overrides C_mult
    down_stride: int
        spatial stride used before increasing number of channels in recursive call
    blocks: int
        number of blocks to recurse for
    training: bool
        whether we are in trining
    """
    logging.info('blocks %d training %s' % (blocks, training))
    assert blocks <= 10
    i = blocks
    in_shape = tf.shape(x)
    if L is None:
        L = C_mult * C
    M = min(C_max, 2 * L)
    if i <= 0:
        key = 'depth%0.2d/more_conv1' % i
        with tf.variable_scope(key):
            set_dict_no_overwrite(variables_dict, key, dict())
            x = conv3_br(x, C=C, L=L, variables_dict=variables_dict[key],
                         training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)
        key = 'depth%0.2d/more_conv2' % i
        with tf.variable_scope(key):
            set_dict_no_overwrite(variables_dict, key, dict())
            x = conv3_br(x, C=L, variables_dict=variables_dict[key], training=training,
                         bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)
        return x

    key = 'depth%0.2d/more_conv1' % i  # C -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=C, L=L, variables_dict=variables_dict[key],
                     training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)

    key = 'depth%0.2d/more_conv2' % i  # L -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=L, L=L, variables_dict=variables_dict[key], training=training,
                     bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)

    x_into = x

    key = 'depth%0.2d/conv_down' % i  # L -> L, reduce spatial resolution
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=L, L=L, kernel_size=down_stride, strides=(1, down_stride, down_stride, down_stride, 1),
                     variables_dict=variables_dict[key], training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)

    # L -> M
    x = allen_sub_u_net(x, variables_dict, C=L, L=M, C_max=C_max, down_stride=down_stride, blocks=i - 1,
                        training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)
    # output of allen_sub_u_net call has M channels

    key = 'depth%0.2d/convt' % i  # M -> L, recover spatial resolution, and reduce channels
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())

        up_shape = tf.concat([in_shape[:4], [L]], axis=0)
        x = conv3t_br(x, C=L, L=M,
                      output_shape=up_shape,
                      kernel_size=down_stride,
                      strides=(1, down_stride, down_stride, down_stride, 1),
                      variables_dict=variables_dict[key], training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)
    x_up = x
    x = tf.concat([x_into, x_up], axis=4)  # NDHWC, results in in 2 * L channels

    key = 'depth%0.2d/less_conv1' % i  # 2 * L -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=2 * L, L=L, variables_dict=variables_dict.get(key, dict()),
                     training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)

    key = 'depth%0.2d/less_conv2' % i  # L -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=L, L=L, variables_dict=variables_dict.get(key, dict()),
                     training=training, bn_momentum=bn_momentum, arch_args_dict=arch_args_dict)
    return x


def conv3t_br(x,
              C,  # C is the output size, so as to pair up with conv
              L,  # L is the input size, so as to pair up with conv
              output_shape,
              kernel_size,
              strides,
              padding='SAME',
              data_format='NDHWC',
              dilations=(1, 1, 1, 1, 1),
              variables_dict=None,
              training=True,
              bn_momentum=0.99,  # tf default
              arch_args_dict=None,
              ):
    """
    Wrapper around conv3d_transpose, batch_normalization, relu
    """
    if arch_args_dict is None:
        arch_args_dict = dict()
    split_bnum = arch_args_dict.get('split_bnum', 1)
    dtype = arch_args_dict.get('weight_dtype', x.dtype)
    normalization_str = arch_args_dict.get('normalization', 'batchnorm')
    logging.info('conv training %s. norm: %s' % (training, normalization_str))

    assert data_format == 'NDHWC'
    ks = kernel_size
    w_xavier = tf.initializers.truncated_normal(0, stddev=np.sqrt(2. / (ks * ks * ks * L)), dtype=dtype)  # different
    w_init = w_xavier
    bn_gamma_ones = tf.initializers.ones(dtype)
    bn_gamma_init = bn_gamma_ones

    w = tf.get_variable(name='w', shape=[ks, ks, ks, C, L], dtype=dtype, initializer=w_init)
    if arch_args_dict.get('kernel_regularizer', None) is not None:
        regularizer = arch_args_dict.get('kernel_regularizer', None)
        tf.contrib.layers.apply_regularization(regularizer, weights_list=[w])
    b = tf.get_variable(name='b', shape=[C], dtype=dtype, initializer=tf.initializers.zeros(dtype))

    if split_bnum > 1:
        xx = tf.split(x, split_bnum, axis=0)
        _output_shape = tf.concat([[tf.shape(xx[0])[0]], output_shape[1:]], axis=0)
        xx = [tf.nn.conv3d_transpose(_, tf.cast(w, x.dtype), output_shape=_output_shape, strides=strides,
                                     data_format=data_format, name=None) for _ in xx]
        x = tf.concat(xx, axis=0) + tf.cast(b, x.dtype)
    else:
        x = tf.nn.conv3d_transpose(x, tf.cast(w, x.dtype), output_shape=output_shape, strides=strides,
                                   data_format=data_format, name=None) + tf.cast(b, x.dtype)
    # torch_momemtum = 1- tf_momentum. default torch_momentum is 0.1, meaning tf_momentum of 0.9
    # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/layers/normalization.py
    # https://pytorch.org/docs/stable/nn.html

    if normalization_str == 'batchnorm':
        # container object that allows us to access moving mean, moving variance
        bn_layer = tf.compat.v1.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=bn_gamma_init)
        x = bn_layer.apply(x, training=training)
    elif normalization_str == 'identity':
        pass
    elif normalization_str == 'groupnorm':
        # not compatible with tf.layers.groupnorm. TODO: implement my own
        x = tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(-4, -3, -2),)
        # x = gn_layer.apply(x, trainable=training, groups=4) # Semantics of trainable is not same as training
    else:
        raise ValueError('Unknown normalization: {}'.format(normalization_str))
    x = tf.nn.relu(x)

    if variables_dict is not None:
        variables_dict['w'] = w
        variables_dict['b'] = b
    if normalization_str == 'batchnorm':
        variables_dict['bn_layer'] = bn_layer
        variables_dict['beta'] = bn_layer.beta
        variables_dict['gamma'] = bn_layer.gamma
        variables_dict['moving_mean'] = bn_layer.moving_mean
        variables_dict['moving_variance'] = bn_layer.moving_variance
    return x


def conv3_br(x,
             C=None,
             L=None,
             kernel_size=3,
             strides=(1, 1, 1, 1, 1),
             padding='SAME',
             data_format='NDHWC',
             dilations=(1, 1, 1, 1, 1),
             variables_dict=None,
             training=True,
             do_relu=True,
             bn_momentum=0.99,  # tf default
             dtype=None,
             arch_args_dict=None,
             ):

    ks = kernel_size
    if C is None:
        C = tf.get_shape(x).as_list()[data_format.index('C')]
    if L is None:
        L = C

    if arch_args_dict is None:
        arch_args_dict = dict()
    split_bnum = arch_args_dict.get('split_bnum', 1)
    dtype = arch_args_dict.get('weight_dtype', x.dtype)
    normalization_str = arch_args_dict.get('normalization', 'batchnorm')
    logging.info('conv3br training=%s. norm=%s. L=%d. C=%d' % (training, normalization_str, L, C))
    assert data_format == 'NDHWC'

    w_xavier = tf.initializers.truncated_normal(0, stddev=np.sqrt(2. / (ks * ks * ks * C)), dtype=dtype)  # different
    w_init = w_xavier
    bn_gamma_ones = tf.initializers.ones(dtype)
    bn_gamma_init = bn_gamma_ones

    w = tf.get_variable(name='w', shape=[ks, ks, ks, C, L], dtype=dtype, initializer=w_init)
    if arch_args_dict.get('kernel_regularizer', None) is not None:
        regularizer = arch_args_dict.get('kernel_regularizer', None)
        tf.contrib.layers.apply_regularization(regularizer, weights_list=[w])
    b = tf.get_variable(name='b', shape=[L], dtype=dtype, initializer=tf.initializers.zeros(dtype))

    if split_bnum > 1:
        xx = tf.split(x, split_bnum, axis=0)
        xx = [tf.nn.conv3d(_, tf.cast(w, x.dtype), strides=strides, padding=padding, data_format=data_format,
                           dilations=dilations, name=None) for _ in xx]
        x = tf.concat(xx, axis=0) + tf.cast(b, x.dtype)
    else:
        x = tf.nn.conv3d(x, tf.cast(w, x.dtype), strides=strides, padding=padding, data_format=data_format,
                         dilations=dilations, name=None) + tf.cast(b, x.dtype)

    if normalization_str == 'batchnorm':
        # container object that allows us to access moving mean, moving variance
        bn_layer = tf.compat.v1.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=bn_gamma_init)
        x = bn_layer.apply(x, training=training)
    elif normalization_str == 'identity':
        pass
    elif normalization_str == 'groupnorm':
        x = tf.contrib.layers.group_norm(x, groups=4, reduction_axes=(1, 2, 3),)
        # x = gn_layer.apply(x, trainable=training, groups=4) # Semantics of trainable is not same as training
    else:
        raise ValueError('Unknown normalization: {}'.format(normalization_str))
    if do_relu:
        x = tf.nn.relu(x)

    if variables_dict is not None:
        variables_dict['w'] = w
        variables_dict['b'] = b
        if normalization_str == 'batchnorm':
            variables_dict['bn_layer'] = bn_layer
            variables_dict['beta'] = bn_layer.beta
            variables_dict['gamma'] = bn_layer.gamma
            variables_dict['moving_mean'] = bn_layer.moving_mean
            variables_dict['moving_variance'] = bn_layer.moving_variance
        elif normalization_str == 'groupnorm':
            pass
            # TODO: add useful diagnostics here
    return x
