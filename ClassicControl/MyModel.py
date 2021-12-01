from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense


def myModel(env, name, activation, num_of_layers=1, densities=None):
    # Basic length checking
    assert num_of_layers == len(densities), 'Amount of layer combinations does not match number of densities'\

    # Get input and out sizes
    obs_size = len(env.reset())
    actions = env.action_space.n

    # Create the model
    model = Sequential(name=name)
    model.add(Input(shape=(obs_size,)))
    for i in range(num_of_layers):
        model.add(Dense(densities[i], input_shape=(actions,), kernel_initializer='lecun_normal',
                        activation=activation, name=f'HiddenLayer{i}'))
    model.add(Dense(actions, activation='softmax'))

    # Compile model with optimizer and other params
    model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
    model.summary()

    return model