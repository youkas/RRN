import tensorflow as tf


def get_kernel_function(kernel):
    if kernel is None or kernel == "pearson":
        return pearson_correlation
    if kernel == "t":
        return t_distribution_kernel
    if kernel == "RBF":
        return rbf_kernel
    if kernel == "linear":
        return linear_kernel
    if kernel == "linear_rbf_mixture":
        return linear_rbf_mixture_kernel
    if kernel == "pearson_rbf_mixture":
        return pearson_rbf_mixture_kernel
    raise Exception("Unknown Kernel function")


def pearson_correlation(_x, _y):
    _x = tf.transpose(tf.cast(_x, tf.float32))
    _y = tf.transpose(tf.cast(_y, tf.float32))
    dsize = tf.shape(_x)[-1]
    xy_t = tf.concat([_x, _y], axis=0)
    mean_t = tf.reduce_mean(xy_t, axis=1, keepdims=True)
    cov_t = ((xy_t - mean_t) @ tf.transpose(xy_t - mean_t)) / tf.cast(dsize - 1, tf.float32)
    cov2_t = tf.linalg.diag(1 / tf.sqrt(tf.linalg.diag_part(cov_t)))
    return cov2_t @ cov_t @ cov2_t


def linear_kernel(_x, _y):
    _x = tf.transpose(tf.cast(_x, tf.float32))
    _y = tf.transpose(tf.cast(_y, tf.float32))
    xy_t = tf.concat([_x, _y], axis=0)
    return xy_t @ tf.transpose(xy_t)


def rbf_kernel(_x, _y, gamma=1.):
    _x = tf.cast(_x, tf.float32)
    _y = tf.cast(_y, tf.float32) if _y is not None else None
    xy = _x if _y is None else tf.concat([_x, _y], axis=1)
    xy = tf.transpose(xy)
    dm = distance_mat(xy)
    return tf.exp(-gamma * tf.square(dm))


def t_distribution_kernel(_x, _y, a=1., b=1.):
    _x = tf.cast(_x, tf.float32)
    _y = tf.cast(_y, tf.float32) if _y is not None else None
    xy = _x if _y is None else tf.concat([_x, _y], axis=1)
    xy = tf.transpose(xy)
    dm = distance_mat(xy)
    return a/(a + b*tf.pow(dm, 2))


def pearson_rbf_mixture_kernel(_x, _y):
    return 0.5*(pearson_correlation(_x, _y) + rbf_kernel(_x, _y))


def linear_rbf_mixture_kernel(_x, _y):
    return 0.5*(linear_kernel(_x, _y) + rbf_kernel(_x, _y))


def distance_mat(xy):
    # Euclidean distance
    x_norm_squared = tf.reduce_sum(tf.square(xy), axis=1, keepdims=True)  # shape: (n, 1)
    dm_squared = x_norm_squared - 2 * tf.matmul(xy, xy, transpose_b=True) + tf.transpose(x_norm_squared)  # shape: (n, n)
    # ensure non-negative distances
    dm_squared = tf.maximum(dm_squared, 1e-12)
    dm = tf.sqrt(dm_squared)
    return dm


