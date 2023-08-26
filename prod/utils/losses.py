import numpy as np
import tensorflow as tf

class LossFunction:
    def __init__(self, name):
        self.name = name
        self._losses = []
        self._loss = None

    def __call__(self, *args, **kwargs):
        self._loss = self.compute(*args, **kwargs)
        return self._loss

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def update(self):
        self._losses.append(self._loss.numpy())

    def get_losses(self):
        return self._losses

def mean_distance(x, y, kind='MSE'):
    distances = None

    if kind == 'RMSE':
        distances = tf.square(x - y) / tf.square(x)

    if kind == 'Frobenius':
        # esquivalent to MSE+Sqrt
        distances = tf.norm(x - y) #Frobenius

    if kind == 'Norm2':
        # Norm 2
        distances = tf.norm(x - y, axis=1)

    if kind == 'RNorm2':
        distances = tf.norm(x - y, axis=1) / tf.norm(x, axis=1)

    if kind == 'Norm1':
        # Norm 1
        distances = tf.norm(x - y, ord=1, axis=1)

    if kind == 'RNorm1':
        # Norm 1
        distances = tf.norm(x - y, ord=1, axis=1) / tf.norm(x, ord=1, axis=1)

    if kind == 'NormInf':
        distances = tf.norm(x - y, ord=np.Inf, axis=1)

    if kind == 'RNormInf':
        # Norm 1
        distances = tf.norm(x - y, ord=np.Inf, axis=1) / tf.norm(x, ord=np.Inf, axis=1)

    if distances is None: #default MSE
        distances = tf.square(x - y)

    return tf.reduce_mean(distances)

class XReconstructionLoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self, 'X Reconstruction')

    def compute(self, x, x_hat):
        return mean_distance(x, x_hat)

class YReconstructionLoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self, 'Y Reconstruction')

    def compute(self, x, y, w, z, w_hat, x_hat, y_hat):
        return mean_distance(y, y_hat)

class RegularizationLoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self, 'Regularization Loss')

    def compute(self, x, y, w, z, w_hat, x_hat, y_hat):
        return self._regularization_loss_(z)

    def _regularization_loss_(self, Z):
        # distances = tf.norm(Z, ord=2, axis=1)
        # distance = tf.reduce_mean(distances)
        distances = tf.norm(Z, ord=np.Inf, axis=1)
        distance = tf.reduce_sum(tf.abs(tf.minimum(1 - distances, 0)))
        return distance

class CalibrationLoss(LossFunction):
    def __init__(self, kind='NormInf'):
        LossFunction.__init__(self, 'Calibration')
        self.kind = kind

    def compute(self, err, err_hat):
        return mean_distance(err, err_hat, kind=self.kind)

class CorrelationLoss(LossFunction):
    def __init__(self):
        LossFunction.__init__(self, 'Correlation Loss')

    def compute(self, x, y, w, z, w_hat, x_hat, y_hat):
        x_cor = self._multiple_correlations_(x, y)
        z_cor = self._multiple_correlations_(z, y)
        return tf.reduce_mean(tf.square(x_cor - z_cor))

    def _multiple_correlations_(self, Z, Y):
        dims = tf.shape(Z)[1]
        cor_mat = np.corrcoef(np.hstack((Z, Y)), rowvar=False)
        cor_xx = cor_mat[:dims, :dims]
        cor_xy = cor_mat[:dims, dims:]
        multiple_correlation = np.array(np.diag(np.matmul(cor_xy.T, np.matmul(np.linalg.pinv(cor_xx), cor_xy))), dtype='float32')
        return tf.convert_to_tensor(multiple_correlation)

    def _correlations_(self, Z, Y):
        corr = np.array([[np.corrcoef(Z[:, i], Y[:, j])[0, 1] for i in range(tf.shape(Z)[1])]
                      for j in range(tf.shape(Y)[1])], dtype='float32')
        return 1. - tf.reduce_mean(tf.abs(tf.convert_to_tensor(corr)))

class SumLoss(LossFunction):
    def __init__(self, *loss_function):
        LossFunction.__init__(self, 'Sum Loss')
        self._loss_functions = list(loss_function)

    def append(self, loss_function):
        self._loss_functions.append(loss_function)

    def compute(self, *args, **kwargs):
        return sum([lf(*args, **kwargs) for lf in self._loss_functions])

    def update(self):
        [lf.update() for lf in self._loss_functions]
        super().update()

    def get_losses(self):
        l = {self.name:self._losses}
        l.update({lf.name:lf.get_losses() for lf in self._loss_functions})
        return l

class AverageLoss(SumLoss):
    def __init__(self, *loss_function):
        SumLoss.__init__(self, *loss_function)
        self.name = 'Average Loss'

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs)/len(self._loss_functions)

class UpdateLoss(tf.keras.callbacks.Callback):
    def __init__(self, tloss, vloss):
        super().__init__()
        self.tloss = tloss
        self.vloss = vloss

    def on_epoch_end(self, epoch, logs=None):
        self.tloss.update()
        self.vloss.update()

