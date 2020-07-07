import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Dense, Flatten, Input,Lambda, Subtract, Concatenate, DepthwiseConv2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import config as cfg





def create_critic_model():

    board_input = Input(shape=(cfg.BOARD_SIZE, cfg.BOARD_SIZE, 3))
    scalar_input = Input(shape=(2,))

    conv1 = Conv2D(64, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(board_input)
    conv2 = Conv2D(32, (3, 3), strides=1,  activation='relu', use_bias=True, padding="valid")(conv1)
    board_flat = Flatten()(conv2)

    scalar_dense1 = Dense(4)(scalar_input)
    scalar_dense2 = Dense(4)(scalar_dense1)

    combined = Concatenate(axis=1)([board_flat, scalar_dense2])

    dense1 = Dense(16)(combined)
    dense2 = Dense(8)(dense1)

    value = Dense(1, activation='tanh')(dense2)

    model = Model(inputs=[board_input, scalar_input], outputs=value)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', experimental_run_tf_function=False)
    model.summary()
    return model
