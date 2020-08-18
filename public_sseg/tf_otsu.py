import tensorflow as tf

def tf_otsu(x,): # x has shape [batch_size, flattened_activations]
    B, A = tf.unstack(tf.shape(x))
    z_fwd = tf.sort(x, axis=-1, direction='ASCENDING')  # numerical precision considerations
    var_fwd = tf_cumulative_var(z_fwd, axis=-1)

    z_rev = tf.reverse(z_fwd, axis=[-1])
    var_rev = tf_cumulative_var(z_rev, axis=-1)

    p_fwd = tf.cast(tf.range(1, 1 + A), dtype=tf.float32) / tf.cast(A, tf.float32)
    q_fwd = 1. - p_fwd
    var_both = p_fwd * var_fwd + q_fwd * tf.reverse(var_rev, axis=[-1])
    z_am = tf.argmin(var_both, axis=-1)
    # Alternative is to use gather_nd on something like tf.stack([tf.range(B), z_am], axis=1)
    gathered = tf.gather(params=z_fwd, indices=z_am, axis=-1) # shape is [B, B]
    return tf.linalg.tensor_diag_part(gathered)



def tf_cumulative_var(z, axis=-1):
    z2 = z ** 2
    cu_sz = tf.cumsum(z, axis=axis)
    cu_sz2 = tf.cumsum(z2, axis=axis)
    n = tf.cast(tf.range(tf.shape(z)[-1]), dtype=tf.float32) + 1
    cu_mz = cu_sz / n
    cu_mz2 = cu_sz2 / n
    cu_var = cu_mz2 - (cu_mz ** 2)
    return cu_var