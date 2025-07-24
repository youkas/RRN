import copy
import os
import warnings

import numpy as np
import tensorflow as tf
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.core.problem import Problem


class RRN(tf.keras.Model):
    """
    Re-parametrization & Regression Networks
    """
    def __init__(self, primary_dimension=None, secondary_dimension=None,
                 encoder=None, surrogate=None, decoder=None, **kwargs):
        super().__init__()
        self.parts = dict(encoder=encoder, surrogate=surrogate, decoder=decoder)
        self.params = {"p_dim": primary_dimension, "s_dim": secondary_dimension,
                       "penalization": kwargs.pop("penalization", 1.), "radius": kwargs.pop("radius", 1.),
                       "kernel": kwargs.pop("kernel", "RBF"), 'parts': list(self.parts.keys())}
        self.loss_function = loss_function
        self._e_optimizer = None
        self._s_optimizer = None
        self._d_optimizer = None

    def encode(self, x):
        return self.encoder(x)

    def predict(self, z, w):
        return self.surrogate(np.hstack((z, w)))

    def decode(self, z, use_model=True, x_lower=None, x_upper=None,
               pop_size=100, n_gen=200, verbose=False, seed=1):
        if self.decoder is not None and use_model:
            return self.decoder(z)
        if len(z.shape) == 1:
            z = np.reshape(z, (-1, 1))
        if x_lower is None or x_upper is None:
            raise Exception('Upper and Lower bounds are required for decoding optimization.')

        decoder = OptDecoder(self.encode, x_lower, x_upper,
                             verbose=verbose, pop_size=pop_size, n_gen=n_gen, seed=seed)
        return np.array([decoder.decode(zi) for zi in z])

    def save(self, filepath):
        if filepath is None:
            return

        for n, m in self.parts.items():
            os.makedirs(filepath + '\\' + n, exist_ok=True)
            if m is not None:
                m.save(filepath + '\\' + n + ".keras")

        np.savez_compressed(filepath + '\\params.npz', **self.params)

    def load(self, filepath):
        keys = ["p_dim", "s_dim", "penalization", "radius", "kernel", 'parts']
        params = np.load(filepath + '\\params.npz', allow_pickle=True)
        self.params = {k: params[k] for k in keys}
        self.parts = {}
        for n in self.params['parts']:
            model_file_name = filepath + '/' + n + ".keras"
            try:
                self.parts[n] = tf.keras.models.load_model(model_file_name)
            except Exception as e:
                self.parts[n] = None
                warnings.warn(str(e))

    def compile(self, **kwargs):
        kwargs["run_eagerly"] = True
        #kwargs["loss"] = self.loss_function
        self._e_optimizer = kwargs.pop("optimizer")
        self._d_optimizer = copy.deepcopy(self._e_optimizer)
        self._s_optimizer = copy.deepcopy(self._e_optimizer)
        super().compile(**kwargs)
        for _, m in self.parts.items():
            if m is not None:
                tf.keras.Model.compile(m, **kwargs)

    def get_weights(self):
        return {k: v.get_weights() if v is not None else None for k, v in self.parts.items()}

    def set_weights(self, weights):
        for k, v in self.parts.items():
            if v is not None:
                v.set_weights(weights[k])

    def _set_history(self, history):
        self.params['history'] = history

    def train_step(self, data):
        with tf.GradientTape() as e_tape, tf.GradientTape() as s_tape, tf.GradientTape() as d_tape:
            loss = self.__process__(data[0], training=True)
            e_loss = loss['correlation'] + loss['regularization']
            s_loss = loss['y_reconstruction']
            t_loss = e_loss + s_loss
            if self.decoder is not None:
                t_loss = t_loss + loss['x_reconstruction']

        e_grads = e_tape.gradient(t_loss, self.encoder.trainable_weights)
        self._e_optimizer.apply_gradients(zip(e_grads, self.encoder.trainable_weights))

        s_grads = s_tape.gradient(t_loss, self.surrogate.trainable_weights)
        self._s_optimizer.apply_gradients(zip(s_grads, self.surrogate.trainable_weights))

        if self.decoder is not None:
            d_grads = d_tape.gradient(t_loss, self.decoder.trainable_weights)
            self._d_optimizer.apply_gradients(zip(d_grads, self.decoder.trainable_weights))

        loss['loss'] = t_loss
        return loss

    def test_step(self, data):
        return self.__process__(data[0], training=False)

    def __process__(self, data, training=False, loss=True, *args, **kwargs):
        xw, y = data
        x, w = xw[:, :self.p_dim], xw[:, self.p_dim:]
        z = self.encoder(x, training=training)
        zw = tf.concat([z, w], axis=1)
        y_hat = self.surrogate(zw, training=training)
        x_hat = None if self.decoder is None else self.decoder(z, training=training)
        if loss:
            kwargs = {'in': {'xw': xw,  'x': x, 'w': w, 'y': y},
                      'out': {'zw': zw, 'z': z, 'x': x_hat, 'y': y_hat},
                      'penalization': self.params["penalization"],
                      'radius': self.params["radius"],
                      'kernel': self.params["kernel"],
                      }
            return self.loss_function(**kwargs)
        return y_hat

    encoder = property(lambda self: self.parts["encoder"])
    surrogate = property(lambda self: self.parts["surrogate"])
    decoder = property(lambda self: self.parts["decoder"])
    p_dim = property(lambda self: self.params["p_dim"])
    s_dim = property(lambda self: self.params["s_dim"])


def loss_function(**kwargs):
    xw = kwargs['in'].get('xw', None)
    x = kwargs['in'].get('x', None)
    w = kwargs['in'].get('w', None)
    y = kwargs['in'].get('y', None)
    zw = kwargs['out'].get('zw', None)
    z = kwargs['out'].get('z', None)
    x_hat = kwargs['out'].get('x', None)
    y_hat = kwargs['out'].get('y', None)
    kernel = kwargs.get("kernel", "RBF")
    penalization = kwargs.get("penalization", 1.)
    radius = kwargs.get("radius", 1.)

    y_reconstruction = tf.reduce_mean(tf.square(y_hat - y))
    x_reconstruction = tf.reduce_mean(tf.square(x_hat - x)) if x_hat is not None else None
    x_corr = correlation(xw, y, kernel=kernel)
    z_corr = correlation(zw, y, kernel=kernel)
    corr = tf.reduce_mean(tf.square(x_corr - z_corr))
    reg = penalization * tf.reduce_mean(tf.maximum(tf.norm(z, ord=np.inf, axis=0) - radius, 0))

    loss = y_reconstruction + corr + reg
    out = {'loss': loss, 'y_reconstruction': y_reconstruction, 'correlation': corr, 'regularization': reg}

    if x_hat is not None:
        loss = loss + x_reconstruction
        out['loss'] = loss
        out['x_reconstruction'] = x_reconstruction
    return out


def pearson_correlation(_x, _y=None):
    _x = tf.transpose(tf.cast(_x, tf.float32))
    _y = tf.transpose(tf.cast(_y, tf.float32)) if _y is not None else None
    dsize = tf.shape(_x)[-1]
    xy_t = _x if _y is None else tf.concat([_x, _y], axis=0)
    mean_t = tf.reduce_mean(xy_t, axis=1, keepdims=True)
    cov_t = ((xy_t - mean_t) @ tf.transpose(xy_t - mean_t)) / tf.cast(dsize - 1, tf.float32)
    cov2_t = tf.linalg.diag(1 / tf.sqrt(tf.linalg.diag_part(cov_t)))
    return cov2_t @ cov_t @ cov2_t


def distance_mat(xy):
    # Euclidean distance
    x_norm_squared = tf.reduce_sum(tf.square(xy), axis=1, keepdims=True)  # shape: (n, 1)
    dm_squared = x_norm_squared - 2 * tf.matmul(xy, xy, transpose_b=True) + tf.transpose(x_norm_squared)  # shape: (n, n)
    # ensure non-negative distances
    dm_squared = tf.maximum(dm_squared, 1e-12)
    dm = tf.sqrt(dm_squared)
    return dm


def rbf(_x, _y, gamma=1.):
    _x = tf.cast(_x, tf.float32)
    _y = tf.cast(_y, tf.float32) if _y is not None else None
    xy = _x if _y is None else tf.concat([_x, _y], axis=1)
    xy = tf.transpose(xy)
    dm_squared = distance_mat(xy)
    return tf.exp(-gamma * tf.sqrt(dm_squared))


def correlation(_x, _y, kernel=None):
    if kernel is None or kernel == "pearson":
        kernel_function = pearson_correlation
    elif kernel == "RBF":
        kernel_function = rbf
    else:
        raise Exception("unknown Kernel function")

    correlation_matrix = kernel_function(_x, _y)
    dims = tf.shape(_x)[1]
    c_xx = correlation_matrix[:dims, :dims]
    c_xy = correlation_matrix[:dims, dims:]
    return tf.matmul(tf.transpose(c_xy), tf.matmul(tf.linalg.pinv(c_xx), c_xy))


class DecoderProblem(Problem):
    def __init__(self, encoder, target, lower, upper):
        super().__init__(n_var=len(lower), n_obj=1, n_constr=0, xl=lower, xu=upper,
                         elementwise_evaluation=False)
        self._encoder = encoder
        self._target = target

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.linalg.norm(self._encoder(x) - self._target, axis=1)


class OptDecoder:
    def __init__(self, handler, x_lower, x_upper, verbose=False, pop_size=100, n_gen=1000, seed=1):
        self._handler = handler
        self._x_lower = x_lower
        self._x_upper = x_upper
        self._verbose = verbose
        self._pop_size = pop_size
        self._n_gen = n_gen
        self._seed = seed

    def decode(self, target):
        algo = PSO(pop_size=self._pop_size)
        res = minimize(DecoderProblem(self._handler, target, self._x_lower, self._x_upper),
                       algo,
                       seed=self._seed,
                       verbose=self._verbose)
        if self._verbose:
            print(f'Fitness {res.F}')
        return np.ravel(res.X)
