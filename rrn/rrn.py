import copy
import os

import numpy as np
import tensorflow as tf
import utils
from rrn.kernel import get_kernel_function


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

    def predict(self, z, w=None):
        return self.surrogate(np.hstack((z, w)) if w is not None else z)

    def decode(self, z, use_model=True, x_lower=None, x_upper=None,
               pop_size=100, n_gen=200, verbose=False, seed=1):
        if use_model:
            if self.decoder is None:
                raise Exception(f"Decoder model is not defined!")
            return self.decoder(z)
        if len(z.shape) == 1:
            z = np.reshape(z, (-1, 1))
        if x_lower is None or x_upper is None:
            raise Exception('Upper and Lower bounds are required for decoding optimization.')

        decoder = utils.OptDecoder(self.encode, x_lower, x_upper,
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

    def compile(self, **kwargs):
        kwargs["run_eagerly"] = True
        # kwargs["loss"] = self.loss_function
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
            kwargs = {'in': {'xw': xw, 'x': x, 'w': w, 'y': y},
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


def correlation(_x, _y, kernel=None):
    kernel_function = get_kernel_function(kernel)

    correlation_matrix = kernel_function(_x, _y)
    dims = tf.shape(_x)[1]
    c_xx = correlation_matrix[:dims, :dims]
    c_xy = correlation_matrix[:dims, dims:]
    return tf.matmul(tf.transpose(c_xy), tf.matmul(tf.linalg.pinv(c_xx), c_xy))
