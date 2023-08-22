import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from prod.utils.losses import *

class RRN:
    def __init__(self, dimension=None, encoder=None, regressor=None, calibration=None, adapter=None, calibrate=True):
        self.history = {}
        self.dimension = dimension
        self.encoder = encoder
        self.regressor = regressor
        self.calibration = calibration
        self.adapter = adapter
        self.calibrate = calibrate
        self.enable_calibration = True

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, z, x_lower, x_upper, verbose=False):
        #from pymoo.algorithms.so_pso import PSO
        from pymoo.algorithms.so_genetic_algorithm import GA
        from pymoo.optimize import minimize
        from pymoo.model.problem import Problem

        class DecoderProblem(Problem):
            def __init__(self, encoder, target, lower, upper):
                super().__init__(n_var=len(lower), n_obj=1, n_constr=0, xl=lower, xu=upper,
                                 elementwise_evaluation=False)
                self._encoder = encoder
                self._target = target

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.linalg.norm(self._encoder(X) - self._target, axis=1)

        #algo = PSO()
        algo = GA(pop_size=20*len(x_lower), eliminate_duplicates=True)
        res = minimize(DecoderProblem(self.encode, z, x_lower, x_upper),
                       algo,
                       seed=1,
                       verbose=False)
        if verbose:
            print(f'Fitness {res.F}')
        return res.X.reshape((1, -1))

    def predict(self, z):
        y_hat = self.regressor.predict(z)
        cal = 0.
        if self.calibrate and self.enable_calibration:
            cal = self.calibration.predict(z)
        c_y = y_hat + cal
        return self.adapter.predict(c_y)

    def save(self, filepath):
        if filepath is None:
            return

        for n, m in self.__get_models__().items():
            m.save(filepath+'\\'+n)

        np.savez_compressed(filepath+'\\params.npz',
                            dimension=self.dimension,
                            enable_calibration=self.enable_calibration,
                            history=self.history)

    def load(self, filepath):
        ms = {n:tf.keras.models.load_model(filepath+'\\'+n) for n in self.__get_models__().keys()}
        self.__set_models__(dict(ms))
        params =  np.load(filepath+'\\params.npz', allow_pickle=True)
        self.dimension = params['dimension']
        self.enable_calibration = params['enable_calibration'] if 'enable_calibration' in params else True
        self.history = params['history'].item()

    def get_weights(self):
        return dict(encoder=self.encoder.get_weights(),
                    regressor=self.regressor.get_weights(),
                    calibration=self.calibration.get_weights(),
                    adapter=self.adapter.get_weights())

    def set_weights(self, weights):
        self.encoder.set_weights(weights['encoder'])
        self.regressor.set_weights(weights['regressor'])
        self.calibration.set_weights(weights['calibration'])
        self.adapter.set_weights(weights['adapter'])

    def __get_models__(self):
        return {'encoder':self.encoder, 'regressor':self.regressor, 'calibration':self.calibration, 'adapter':self.adapter}

    def __set_models__(self, models):
        self.encoder = models['encoder']
        self.regressor = models['regressor']
        self.calibration = models['calibration']
        self.adapter = models['adapter']

class Trainer(tf.keras.Model):
    def __init__(self, model, x_dim, w_dim, training_mode='FitAndCalibrate'):
        tf.keras.Model.__init__(self)
        assert isinstance(model, RRN)
        self.model = model
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.training_loss = None
        self.validation_loss = None
        self.cal_training_loss = None
        self.cal_validation_loss = None
        self.optimizer = None
        self._training_mode = training_mode
        self._mode = None
        self.stop_epoch = None

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def compile(self, learning_rate=0.01, **kwargs):
        tf.keras.Model.compile(self, run_eagerly=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.training_loss = SumLoss(YReconstructionLoss(), RegularizationLoss(), CorrelationLoss())
        self.validation_loss = SumLoss(YReconstructionLoss(), RegularizationLoss(), CorrelationLoss())

        self.cal_training_loss = SumLoss(CalibrationLoss(kind='NormInf')) # NormInf or Norm1
        self.cal_validation_loss = SumLoss(CalibrationLoss(kind='NormInf'))

    def train(self, x, y, val_x, val_y, epochs=10, calibration_epochs=None, batch_size=50, verbose=0, model_path=None, patience=5):
        self.stop_epoch = None
        if self._training_mode == 'FitAndCalibrate':
            self.model.enable_calibration = True
            print('Fitting And Calibration')
            self._fit_and_calibrate_([x, y], [val_x, val_y], epochs=epochs, batch_size=batch_size,
                                     verbose=verbose, patience=patience)
        elif self._training_mode == 'FitThenCalibrate':
            self.model.enable_calibration = True
            print('Fitting')
            self._fit_([x, y], [val_x, val_y], epochs=epochs, batch_size=batch_size, verbose=verbose, patience=patience)
            print('Calibrating')
            self._calibrate_([x, y], [val_x, val_y], epochs=calibration_epochs if calibration_epochs is not None else epochs, batch_size=batch_size, verbose=verbose, patience=patience)
        elif self._training_mode == 'Calibrate':
            print('Calibrating')
            self.model.enable_calibration = True
            self._calibrate_([x, y], [val_x, val_y], epochs=calibration_epochs if calibration_epochs is not None else epochs, batch_size=batch_size, verbose=verbose, patience=patience)
        elif self._training_mode == 'Fit':
            self.model.enable_calibration = False
            print('Fitting')
            self._fit_([x, y], [val_x, val_y], epochs=epochs, batch_size=batch_size, verbose=verbose, patience=patience)
        else:
            raise Exception(f'Unknown Training mode: {self._training_mode}')

        if 'fitting' in self.model.history:
            self.stop_epoch = len(self.model.history['fitting']['training']['Sum Loss'])

        if model_path is not None:
            self.model.save(model_path)


    def test(self, x, y):
        if not len(x) or not len(y):
            return

        z = self.model.encode(x[:, :self.x_dim])
        zw = np.concatenate([z, x[:, self.x_dim:]], axis=1)
        y_hat = self.model.predict(zw)

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

    def plot_params(self, x, y):
        def _plot_params_(p, title):
            # plot
            fig, ax = plt.subplots()

            for i in range(p.shape[1]):
                ax.scatter(p[:, i], np.ones(len(p)) + i)
            plt.title(title)
            plt.show()

        z = self.encode(x)
        x_hat, y_hat = self.predict(z)
        _plot_params_(z, 'lattent space')
        _plot_params_(x_hat, 'Reconstructed X')

    def _fit_and_calibrate_(self, data, val_data, epochs=10, batch_size=50, verbose=0, patience=5):
        self._mode = 'Fit'
        self.fit(data,
                           validation_data=val_data,
                           epochs=epochs,
                           verbose=verbose,
                           batch_size=batch_size,
                           callbacks=[UpdateLoss(self.training_loss, self.validation_loss),
                                      UpdateLoss(self.cal_training_loss, self.cal_validation_loss),
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
        self.model.history['calibration'] = {'training':self.cal_training_loss.get_losses(),
                                             'validation':self.cal_validation_loss.get_losses()}
        self._mode = None

    def _fit_(self, data, val_data, epochs=10, batch_size=50, verbose=0, patience=5):
        self._mode = 'Fit'
        self.fit(data,
                           validation_data=val_data,
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

    def _calibrate_(self, data, val_data, epochs=10, batch_size=50, verbose=0, patience=5):
        self._mode = 'Calibrate'
        self.fit(data,
                 validation_data=val_data,
                 epochs=epochs,
                 verbose=verbose,
                 batch_size=batch_size,
                 callbacks=[UpdateLoss(self.cal_training_loss, self.cal_validation_loss),
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
        self.model.history['calibration'] = {'training':self.cal_training_loss.get_losses(), 'validation':self.cal_validation_loss.get_losses()}
        self._mode = None

    def train_step(self, data):
        x, y = data[0]
        if self._training_mode == 'FitAndCalibrate':
            return self._fitting_and_calibration_(x, y)
        else:
            loss = self._fitting_(x, y) if self._mode == 'Fit' else self._calibration_(x, y)
            return {'loss': loss}

    def test_step(self, data):
        x, y = data[0]
        if self._training_mode == 'FitAndCalibrate':
            return self._test_fitting_and_calibration_(x, y)
        else:
            loss = self._test_fitting_(x, y) if self._mode == 'Fit' else self._test_calibration_(x, y)
            return {'loss': loss}

    def _fitting_and_calibration_(self, x, y):
        # Run forward pass.
        with tf.GradientTape() as e_tape, tf.GradientTape() as d_tape:
            z = self.model.encoder(x[:, :self.x_dim], training=True)
            zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
            y_hat = self.model.regressor(zw, training=True)
            f_error = self.training_loss(x[:, :self.x_dim], y, None, z, None, None, y_hat)

        grads = e_tape.gradient(f_error, self.model.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.encoder.trainable_weights))

        grads = d_tape.gradient(f_error, self.model.regressor.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.regressor.trainable_weights))

        with tf.GradientTape() as c_tape:
            z = self.model.encoder(x[:, :self.x_dim], training=False)
            zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
            y_hat = self.model.regressor(zw, training=False)
            err_hat = self.model.calibration(zw, training=True)
            c_error = self.cal_training_loss(y - y_hat, err_hat)

        grads = c_tape.gradient(c_error, self.model.calibration.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.calibration.trainable_weights))
        return {'fitting':f_error, 'calibration':c_error}

    def _fitting_(self, x, y):
        # Run forward pass.
        with tf.GradientTape() as e_tape, tf.GradientTape() as d_tape:
            z = self.model.encoder(x[:, :self.x_dim], training=True)
            zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
            y_hat = self.model.regressor(zw, training=True)
            error = self.training_loss(x[:, :self.x_dim], y, None, z, None, None, y_hat)

        grads = e_tape.gradient(error, self.model.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.encoder.trainable_weights))

        grads = d_tape.gradient(error, self.model.regressor.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.regressor.trainable_weights))

        return error

    def _calibration_(self, x, y):
        with tf.GradientTape() as c_tape:
            z = self.model.encoder(x[:, :self.x_dim], training=False)
            zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
            y_hat = self.model.regressor(zw, training=False)
            err_hat = self.model.calibration(zw, training=True)
            error = self.cal_training_loss(y - y_hat, err_hat)

        grads = c_tape.gradient(error, self.model.calibration.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.calibration.trainable_weights))

        return error

    def _test_fitting_and_calibration_(self, x, y):
        z = self.model.encoder(x[:, :self.x_dim], training=False)
        zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
        y_hat = self.model.regressor(zw, training=False)
        f_error = self.validation_loss(x[:, :self.x_dim], y, None, z, None, None, y_hat)

        z = self.model.encoder(x[:, :self.x_dim], training=False)
        zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
        y_hat = self.model.regressor(zw, training=False)
        err_hat = self.model.calibration(zw, training=False)
        c_error = self.cal_validation_loss(y - y_hat, err_hat)

        return {'fitting':f_error, 'calibration':c_error}

    def _test_fitting_(self, x, y):
        z = self.model.encoder(x[:, :self.x_dim], training=False)
        zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
        y_hat = self.model.regressor(zw, training=False)
        return self.validation_loss(x[:, :self.x_dim], y, None, z, None, None, y_hat)

    def _test_calibration_(self, x, y):
        z = self.model.encoder(x[:, :self.x_dim], training=False)
        zw = tf.concat([z, x[:, self.x_dim:]], axis=1)
        y_hat = self.model.regressor(zw, training=False)
        err_hat = self.model.calibration(zw, training=False)
        return self.cal_validation_loss(y - y_hat, err_hat)

activation_function = "relu" #swish
#To the purpose of this paper, we the activation function for both the Encoder and the Regressor
#is relu because it is widely used by ANN practioners, but in our numerous results, swish activation function
#showed similar and sometimes improved accuracy nontheless.

class Surrogate(RRN):
    def __init__(self, x_dim, w_dim, y_dim, z_dim,
                 x_dispersion=None, x_middle=None, y_dispersion=None, y_middle=None,
                 encoder_length=50, regressor_length=24, verbose=0):

        self.x_dim = x_dim
        self.w_dim = w_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.x_scale = 1 / x_dispersion if x_dispersion is not None else np.ones(self.x_dim + self.w_dim)
        self.x_offset = - x_middle * self.x_scale if x_middle is not None else np.zeros(self.x_dim + self.w_dim)
        self.scale = y_dispersion if y_dispersion is not None else np.ones(self.y_dim)
        self.offset = y_middle if y_middle is not None else np.zeros(self.y_dim)

        self.encoder_length = encoder_length
        self.regressor_length = regressor_length
        self.verbose = verbose

        assert len(self.x_scale) == self.x_dim + self.w_dim
        assert len(self.x_offset) == self.x_dim + self.w_dim
        assert len(self.scale) == self.y_dim
        assert len(self.offset) == self.y_dim
        RRN.__init__(self,
                     dimension=z_dim,
                     encoder=self.__get_encoder__(),
                     regressor=self.__get_regressor__(),
                     calibration=self.__get_calibration__(),
                     adapter=self.__get_adapter__())

    def __get_encoder__(self):
        x_input = tf.keras.Input(shape=self.x_dim, name='X')

        z = tf.keras.layers.Rescaling(self.x_scale[:self.x_dim], offset=self.x_offset[:self.x_dim])(x_input)

        z = tf.keras.layers.Dense(self.encoder_length, activation=activation_function)(z)
        z = tf.keras.layers.Dense(self.z_dim)(z)

        model = tf.keras.Model(inputs=x_input, outputs=z, name="Encoder")
        if self.verbose:
            model.summary()
        return model

    def __get_regressor__(self):
        zw_input = tf.keras.Input(shape=self.z_dim + self.w_dim, name='ZWLayer')

        z_w_input = tf.split(zw_input, [self.z_dim, self.w_dim], 1)
        z = tf.keras.layers.Rescaling(self.x_scale[self.x_dim:], offset=self.x_offset[self.x_dim:])(z_w_input[1])
        z = tf.keras.layers.Concatenate()([z_w_input[0], z])

        subnets = [self._get_Cl_net_(z), self._get_Cd_net_(z)]
        y_hat = tf.keras.layers.Concatenate()(subnets)

        model = tf.keras.Model(inputs=zw_input, outputs=y_hat, name="Decoder")
        if self.verbose:
            model.summary()
        return model

    def __get_calibration__(self):
        zw_input = tf.keras.Input(shape=self.z_dim + self.w_dim, name='ZWLayer')

        z_w_input = tf.split(zw_input, [self.z_dim, self.w_dim], 1)
        z = tf.keras.layers.Rescaling(self.x_scale[self.x_dim:], offset=self.x_offset[self.x_dim:])(z_w_input[1])
        z = tf.keras.layers.Concatenate()([z_w_input[0], z])

        subnets = [self._get_Cl_net_(z), self._get_Cd_net_(z)]
        y_hat = tf.keras.layers.Concatenate()(subnets)

        model = tf.keras.Model(inputs=zw_input, outputs=y_hat, name="Calibration")
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
