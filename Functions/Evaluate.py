import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()

# from keras import layers
# from keras.layers import Input, Conv2D
# from keras.models import Model, load_model

class Evaluate():
    def __init__(self):
        pass

    def forward_keras(self, weights, inputs):
        # model = define_keras_model((1000, 1500, 18))
        # assign_weights(model, weights)
        model = load_model('HDRMapping.h5')
        # print(model.summary())
        output = model.evaluate(x=inputs)
        print(output.shape)

    def forward_tensorflow(self, network, inputs):
        # inputs = np.expand_dims(inputs, 0)
        inputs = np.reshape(inputs, [-1, 1000, 1500, 18])
        parameters = create_tensor_parameters(network)
        # parameters = random_init()

        print_all(inputs, parameters)
        # plt.imshow(inputs[0, :, :, 0:3])
        # plt.show()

        x = tf.placeholder(tf.float64, shape=(None, 1000, 1500, 18))
        output = forward_pass(inputs, parameters)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        final_output = sess.run(output, feed_dict={x: inputs})
        print(final_output)


def create_tensor_parameters(network):
    parameters = {
        'W1': tf.cast(tf.convert_to_tensor(network['layer_1']['weights']['filters']), tf.float64),
        'B1': tf.cast(tf.convert_to_tensor(network['layer_1']['weights']['bias'].reshape(100)), tf.float64),
        'W2': tf.cast(tf.convert_to_tensor(network['layer_2']['weights']['filters']), tf.float64),
        'B2': tf.cast(tf.convert_to_tensor(network['layer_2']['weights']['bias'].reshape(100)), tf.float64),
        'W3': tf.cast(tf.convert_to_tensor(network['layer_3']['weights']['filters']), tf.float64),
        'B3': tf.cast(tf.convert_to_tensor(network['layer_3']['weights']['bias'].reshape(50)), tf.float64),
        'W4': tf.cast(tf.convert_to_tensor(network['layer_4']['weights']['filters']), tf.float64),
        'B4': tf.cast(tf.convert_to_tensor(network['layer_4']['weights']['bias'].reshape(9)), tf.float64)
    }

    # parameters = {
    #     'W1': network['layer_1']['weights']['filters'],
    #     'B1': network['layer_1']['weights']['bias'].reshape(100),
    #     'W2': network['layer_2']['weights']['filters'],
    #     'B2': network['layer_2']['weights']['bias'].reshape(100),
    #     'W3': network['layer_3']['weights']['filters'],
    #     'B3': network['layer_3']['weights']['bias'].reshape(50),
    #     'W4': network['layer_4']['weights']['filters'],
    #     'B4': network['layer_4']['weights']['bias'].reshape(9)
    # }

    return parameters


def random_init():
    parameters = {
        'W1': tf.Variable(tf.random_normal([7, 7, 18, 100])),
        'B1': tf.Variable(tf.zeros([100])),
        'W2': tf.Variable(tf.random_normal([5, 5, 100, 100])),
        'B2': tf.Variable(tf.zeros([100])),
        'W3': tf.Variable(tf.random_normal([3, 3, 100, 50])),
        'B3': tf.Variable(tf.zeros([50])),
        'W4': tf.Variable(tf.random_normal([1, 1, 50, 9])),
        'B4': tf.Variable(tf.zeros([9]))
    }

    return parameters


def print_all(inputs, parameters):
    print('inputs shape: ' + str(inputs.shape) + ' , dtype: ' + str(inputs.dtype))
    print(np.isnan(inputs.any()))
    print('W1: ' + str(parameters['W1'].shape) + ' , dtype: ' + str(parameters['W1'].dtype))
    print('B1: ' + str(parameters['B1'].shape) + ' , dtype: ' + str(parameters['B1'].dtype))
    print('W4: ' + str(parameters['W4'].shape) + ' , dtype: ' + str(parameters['W4'].dtype))
    print('B4: ' + str(parameters['B4'].shape) + ' , dtype: ' + str(parameters['B4'].dtype))


def forward_pass(parameters, inputs):

    conv = tf.nn.conv2d(inputs, parameters['W1'], strides=[1, 1, 1, 1], padding='SAME')
    conv_bias = tf.nn.bias_add(conv, parameters['B1'])
    layer_1_output = tf.nn.relu(conv_bias)

    layer_2_output = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(layer_1_output, parameters['W2'], strides=[1, 1, 1, 1], padding='VALID'),
            parameters['B2']
        )
    )

    layer_3_output = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(layer_2_output, parameters['W3'], strides=[1, 1, 1, 1], padding='VALID'),
            parameters['B3']
        )
    )

    layer_4_output = tf.nn.sigmoid(
        tf.nn.bias_add(
            tf.nn.conv2d(layer_3_output, parameters['W4'], strides=[1, 1, 1, 1], padding='VALID'),
            parameters['B4']
        )
    )

    return layer_4_output


def define_keras_model(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(100, (7, 7), activation='relu', strides=(1, 1))(X_input)
    X = Conv2D(100, (5, 5), activation='relu', strides=(1, 1))(X)
    X = Conv2D(50, (3, 3), activation='relu', strides=(1, 1))(X)
    X = Conv2D(9, (1, 1), activation='sigmoid', strides=(1, 1))(X)

    model = Model(inputs=X_input, outputs=X, name='HDRMapping')

    return model


def assign_weights(model, weights):
    model.layers[1].set_weights([weights['layer_1']['weights']['filters'], weights['layer_1']['weights']['bias'].reshape(100)])
    model.layers[2].set_weights([weights['layer_2']['weights']['filters'], weights['layer_2']['weights']['bias'].reshape(100)])
    model.layers[3].set_weights([weights['layer_3']['weights']['filters'], weights['layer_3']['weights']['bias'].reshape(50)])
    model.layers[4].set_weights([weights['layer_4']['weights']['filters'], weights['layer_4']['weights']['bias'].reshape(9)])

    model.save('HDRMapping.h5')
