from rrn import RRN
import numpy as np
import tensorflow as tf


def load_data():
    dataset = np.load(f'./data/425 12 +-200 3000 + mi/data.npz')['data']
    x_dim = 12
    w_dim = 2
    y_dim = 2  # Cl, Cd
    return [dataset[:, :x_dim + w_dim], dataset[:, x_dim + w_dim:]], x_dim, w_dim, y_dim


def get_encoder(x_dim, z_dim, activation_function, layout):
    x = tf.keras.Input(shape=x_dim, name='X')
    z = x
    for n in layout:
        z = tf.keras.layers.Dense(n, activation=activation_function)(z)
    y = tf.keras.layers.Dense(z_dim)(z)
    model = tf.keras.Model(inputs=x, outputs=y, name="Encoder")
    model.summary()
    return model


def get_decoder(z_dim, x_dim, activation_function, layout):
    x = tf.keras.Input(shape=z_dim, name='Z')
    z = x
    for n in layout:
        z = tf.keras.layers.Dense(n, activation=activation_function)(z)
    y = tf.keras.layers.Dense(x_dim)(z)
    model = tf.keras.Model(inputs=x, outputs=y, name="Decoder")
    model.summary()
    return model


def get_surrogate(z_dim, w_dim, y_dim, activation_function, layout):
    x = tf.keras.Input(shape=z_dim + w_dim, name='ZWLayer')
    z = x
    for n in layout:
        z = tf.keras.layers.Dense(n, activation=activation_function)(z)

    y = tf.keras.layers.Dense(y_dim)(z)
    model = tf.keras.Model(inputs=x, outputs=y, name="Surrogate")
    model.summary()
    return model


train_data, X_DIMENSION, W_DIMENSION, Y_DIMENSION = load_data()

Z_DIMENSION = 2

print(f"Dimensions (X, W, Y, Z) = ({X_DIMENSION}, {W_DIMENSION}, {Y_DIMENSION}, , {Z_DIMENSION})")

ACTIVATION_FUNCTION = "relu"
ENCODER_LAYOUT = [64, 32]
SURROGATE_LAYOUT = [32, 32, 32]
KERNEL = "RBF"  # "RBF"
RADIUS = 1.
RADIUS_PENALIZATION = 1.
PATIENCE = 5
EPOCHS = 10000

rrn_model = RRN(primary_dimension=X_DIMENSION, secondary_dimension=W_DIMENSION,
                encoder=get_encoder(X_DIMENSION, Z_DIMENSION, ACTIVATION_FUNCTION, ENCODER_LAYOUT),
                surrogate=get_surrogate(Z_DIMENSION, W_DIMENSION, Y_DIMENSION, ACTIVATION_FUNCTION, SURROGATE_LAYOUT),
                decoder=get_decoder(Z_DIMENSION, X_DIMENSION, ACTIVATION_FUNCTION, reversed(ENCODER_LAYOUT)),
                kernel=KERNEL,
                penalization=RADIUS,
                radius=RADIUS_PENALIZATION)

rrn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

rrn_model.fit(train_data, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping])

evaluate_loss = rrn_model.evaluate(train_data, return_dict=True)

rrn_model.save(f"./model/rrn_model/")
print(f"Evaluate Losses: {evaluate_loss}")

loaded_rrn_model = RRN()
loaded_rrn_model.load(f"./model/rrn_model/")
loaded_evaluate_loss = rrn_model.evaluate(train_data, return_dict=True)
print(f"Loaded Evaluate Losses: {evaluate_loss}")

