import random
import gym
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop


def myModel(input_space, action_space, name, opt=str, num_of_layers=1, densities=None):
    # Starting with inputs from available inputs,
    # We really only need one layer with 128 nodes
    # to solve the ClassicControl games.
    assert num_of_layers == len(densities)
    if densities is None:
        densities = [128]

    dataInput = Input(input_space)
    layer = dataInput
    for i in range(num_of_layers):
        layer = Dense(densities[i], input_shape=input_space, activation="relu", kernel_initializer='he_uniform')(layer)
    output = Dense(action_space, activation='linear')(layer)

    model = Model(inputs=dataInput, outputs=output, name=name)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model
