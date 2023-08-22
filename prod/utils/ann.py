import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import Model

from prod.utils.losses import *
import tensorflow as tf

class ANN:
    def __init__(self, surrogate=None, adapter=None):
        self.history = {}
        self.surrogate = surrogate
        self.adapter = adapter

    def predict(self, z):
        y_hat = self.surrogate.predict(z)
        return self.adapter.predict(y_hat)

    def save(self, filepath):
        if filepath is None:
            return
        for n, m in self.__get_models__().items():
            m.save(filepath+'\\'+n)

        np.savez_compressed(filepath+'\\params.npz',
                            history=self.history)

    def load(self, filepath):
        ms = {n:tf.keras.models.load_model(filepath+'\\'+n) for n in self.__get_models__().keys()}
        self.__set_models__(dict(ms))
        params =  np.load(filepath+'\\params.npz', allow_pickle=True)
        self.history = params['history'].item()

    def get_weights(self):
        return dict(surrogate=self.surrogate.get_weights(), adapter=self.adapter.get_weights())

    def set_weights(self, weights):
        self.surrogate.set_weights(weights['surrogate'])
        self.adapter.set_weights(weights['adapter'])

    def __get_models__(self):
        return {'surrogate':self.surrogate, 'adapter':self.adapter}

    def __set_models__(self, models):
        self.surrogate = models['surrogate']
        self.adapter = models['adapter']

class Trainer(tf.keras.Model):
    def __init__(self, model):
        tf.keras.Model.__init__(self)
        assert isinstance(model, ANN)
        self.model = model
        self.training_loss = None
        self.validation_loss = None
        self.optimizer = None
        self.stop_epoch = None

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def compile(self, learning_rate=0.01, **kwargs):
        tf.keras.Model.compile(self, run_eagerly=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.training_loss = SumLoss(YReconstructionLoss())
        self.validation_loss = SumLoss(YReconstructionLoss())

    def train(self, x, y, val_x, val_y, epochs=10, batch_size=50, verbose=0, model_path=None, patience=5):
        self.stop_epoch = None
        print('Fitting')
        self._fit_([x, y], [val_x, val_y], epochs=epochs, batch_size=batch_size, verbose=verbose, patience=patience)
        if 'fitting' in self.model.history:
            self.stop_epoch = len(self.model.history['fitting']['training']['Sum Loss'])
        if model_path is not None:
            self.model.save(model_path)

    def test(self, x, y):
        if not len(x) or not len(y):
            return
        y_hat = self.model.surrogate(x)

        error = {'MPE': np.mean((tf.abs(y - y_hat) / y)),
                 'MSE': np.mean((tf.square(y - y_hat)))}

        print(f'Error : {error}')
        return error

    def plot_history(self, key='fitting'):
        if key not in self.model.history:
            return
        history = self.model.history[key]

        fig, ax = plt.subplots()

        types = ['training', 'validation']
        styles = ['-', '--']
        keys = ["X reconstruction loss", "Y reconstruction loss",
                "Regularization loss", "Correlation loss", "Total loss"]
        keys = history['training'].keys()

        # keys = ["X reconstruction loss", "Y reconstruction loss"]
        colors = ['k', 'r', 'g', 'b', 'c', 'c', 'k']
        # colors = ['b', 'r']
        for i, t in enumerate(types):
            for j, k in enumerate(history[t].keys()):
                ax.plot(np.array(history[t][k]), styles[i], linewidth=0.7, color=colors[j],
                        label=t.capitalize() + ' ' + k)
        ax.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def _fit_(self, data, val_data, epochs=10, batch_size=50, verbose=0, patience=5):
        self._mode = 'Fit'
        self.fit(data, validation_data=val_data,
                       epochs=epochs,
                       verbose=verbose,
                       batch_size=batch_size,
                       callbacks=[UpdateLoss(self.training_loss, self.validation_loss),
                                  EarlyStopping(
                                      monitor='val_loss',
                                      min_delta=0,
                                      patience=patience,
                                      verbose=1,
                                      mode='auto',
                                      baseline=None,
                                      restore_best_weights=True
                                  )
                                  ])


        self.model.history['fitting'] = {'training':self.training_loss.get_losses(),
                                         'validation':self.validation_loss.get_losses()}
        self._mode = None
        print(len(self.model.history['fitting']['training']['Sum Loss']))

    def train_step(self, data):
        x, y = data[0]
        loss = self._fitting_(x, y)
        return {'loss': loss}

    def test_step(self, data):
        x, y = data[0]
        loss = self._test_fitting_(x, y)
        return {'loss': loss}

    def _fitting_(self, x, y):
        # Run forward pass.
        with tf.GradientTape() as e_tape:
            y_hat = self.model.surrogate(x, training=True)
            error = self.training_loss(None, y, None, None, None, None, y_hat)

        grads = e_tape.gradient(error, self.model.surrogate.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.surrogate.trainable_weights))

        return error

    def _test_fitting_(self, x, y):
        y_hat = self.model.surrogate(x, training=False)
        return self.validation_loss(None, y, None, None, None, None, y_hat)

activation_function = "relu" #"swish"

class RRNLike(ANN):
    def __init__(self, x_dim, w_dim, z_dim, y_dim,
                 x_dispersion=None, x_middle=None, y_dispersion=None, y_middle=None,
                 encoder_length=50, regressor_length=24, verbose=0):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.y_dim = y_dim
        self.encoder_length = encoder_length
        self.regressor_length = regressor_length
        self.verbose = verbose

        self.x_scale = 1 / x_dispersion if x_dispersion is not None else np.ones(self.x_dim + self.w_dim)
        self.x_offset = - x_middle * self.x_scale if x_middle is not None else np.zeros(self.x_dim + self.w_dim)
        self.scale = y_dispersion if y_dispersion is not None else np.ones(self.y_dim)
        self.offset = y_middle if y_middle is not None else np.zeros(self.y_dim)

        assert len(self.x_scale) == self.x_dim + self.w_dim
        assert len(self.x_offset) == self.x_dim + self.w_dim
        assert len(self.scale) == self.y_dim
        assert len(self.offset) == self.y_dim
        ANN.__init__(self, surrogate=self.__get_surrogate__(),
                           adapter=self.__get_adapter__())

    def __get_surrogate__(self):
        x_input = tf.keras.Input(shape=self.x_dim + self.w_dim, name='X')

        xw = tf.keras.layers.Rescaling(self.x_scale, offset=self.x_offset)(x_input)
        x_w_input = tf.split(xw, [self.x_dim, self.w_dim], 1)

        #equivalaten to reparametrization layer
        z = tf.keras.layers.Dense(self.encoder_length, activation=activation_function)(x_w_input[0])
        z = tf.keras.layers.Dense(self.z_dim)(z)

        z = tf.keras.layers.Concatenate()([z, x_w_input[1]])

        subnets = [self._get_Cl_net_(z), self._get_Cd_net_(z)]
        y_hat = tf.keras.layers.Concatenate()(subnets)

        model = tf.keras.Model(inputs=x_input, outputs=y_hat, name="Surrogate")
        if self.verbose:
            model.summary()
        return model

    def __get_adapter__(self):
        y_inputs = tf.keras.Input(shape=self.y_dim, name='y_layer')

        y = tf.keras.layers.Rescaling(self.scale, offset=self.offset)(y_inputs)
        model = tf.keras.Model(inputs=y_inputs, outputs=y, name="Adaptation")
        if self.verbose:
            model.summary()
        return model

    def _get_Cl_net_(self, z):
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(z)
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(y_hat)
        return tf.keras.layers.Dense(1)(y_hat)

    def _get_Cd_net_(self, z):
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(z)
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(y_hat)
        return tf.keras.layers.Dense(1)(y_hat)

class Surrogate(ANN):
    def __init__(self, x_dim, w_dim, z_dim, y_dim,
                 x_dispersion=None, x_middle=None, y_dispersion=None, y_middle=None,
                 encoder_length=50, regressor_length=24, verbose=0):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.y_dim = y_dim
        self.encoder_length = encoder_length
        self.regressor_length = regressor_length
        self.verbose = verbose

        self.x_scale = 1 / x_dispersion if x_dispersion is not None else np.ones(self.x_dim + self.w_dim)
        self.x_offset = - x_middle * self.x_scale if x_middle is not None else np.zeros(self.x_dim + self.w_dim)
        self.scale = y_dispersion if y_dispersion is not None else np.ones(self.y_dim)
        self.offset = y_middle if y_middle is not None else np.zeros(self.y_dim)

        assert len(self.x_scale) == self.x_dim + self.w_dim
        assert len(self.x_offset) == self.x_dim + self.w_dim
        assert len(self.scale) == self.y_dim
        assert len(self.offset) == self.y_dim
        ANN.__init__(self, surrogate=self.__get_surrogate__(),
                           adapter=self.__get_adapter__())

    def __get_surrogate__(self):
        x_input = tf.keras.Input(shape=self.x_dim + self.w_dim, name='X')

        xw = tf.keras.layers.Rescaling(self.x_scale, offset=self.x_offset)(x_input)
        subnets = [self._get_Cl_net_(xw), self._get_Cd_net_(xw)]
        y_hat = tf.keras.layers.Concatenate()(subnets)

        model = tf.keras.Model(inputs=x_input, outputs=y_hat, name="Surrogate")
        if self.verbose:
            model.summary()
        return model

    def __get_adapter__(self):
        y_inputs = tf.keras.Input(shape=self.y_dim, name='y_layer')

        y = tf.keras.layers.Rescaling(self.scale, offset=self.offset)(y_inputs)
        model = tf.keras.Model(inputs=y_inputs, outputs=y, name="Adaptation")
        if self.verbose:
            model.summary()
        return model

    def _get_Cl_net_(self, z):
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(z)
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(y_hat)
        return tf.keras.layers.Dense(1)(y_hat)

    def _get_Cd_net_(self, z):
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(z)
        y_hat = tf.keras.layers.Dense(self.regressor_length, activation=activation_function)(y_hat)
        return tf.keras.layers.Dense(1)(y_hat)


