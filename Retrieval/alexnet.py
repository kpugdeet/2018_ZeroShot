################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################
import tensorflow as tf
from numpy import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def alexnet_scratch(inputs, scope='alexnet'):
    net_data = load(open("/media/dataHD3/kpugdeet/PRE/bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    with tf.variable_scope(scope, 'alexnet', [inputs], reuse=None) as sc:

        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1W = tf.get_variable(name='conv1W', shape=net_data["conv1"][0].shape)
        conv1b = tf.get_variable(name='conv1b', shape=net_data["conv1"][1].shape)
        conv1_in = conv(inputs, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1 lrn(2, 2e-05, 0.75, name='norm1')
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool1 max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2 conv(5, 5, 256, 1, 1, group=2, name='conv2')
        conv2W = tf.get_variable(name='conv2W', shape=net_data["conv2"][0].shape)
        conv2b = tf.get_variable(name='conv2b', shape=net_data["conv2"][1].shape)
        conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2 lrn(2, 2e-05, 0.75, name='norm2')
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool2 max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3 conv(3, 3, 384, 1, 1, name='conv3')
        conv3W = tf.get_variable(name='conv3W', shape=net_data["conv3"][0].shape)
        conv3b = tf.get_variable(name='conv3b', shape=net_data["conv3"][1].shape)
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
        conv3 = tf.nn.relu(conv3_in)

        # conv4 conv(3, 3, 384, 1, 1, group=2, name='conv4')
        conv4W = tf.get_variable(name='conv4W', shape=net_data["conv4"][0].shape)
        conv4b = tf.get_variable(name='conv4b', shape=net_data["conv4"][1].shape)
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
        conv4 = tf.nn.relu(conv4_in)

        # conv5 conv(3, 3, 256, 1, 1, group=2, name='conv5')
        conv5W = tf.get_variable(name='conv5W', shape=net_data["conv5"][0].shape)
        conv5b = tf.get_variable(name='conv5b', shape=net_data["conv5"][1].shape)
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5 max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fc6 fc(4096, name='fc6')
        fc6W = tf.get_variable(name='fc6W', shape=net_data["fc6"][0].shape)
        fc6b = tf.get_variable(name='fc6b', shape=net_data["fc6"][1].shape)
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

        # fc7 fc(4096, name='fc7')
        fc7W = tf.get_variable(name='fc7W', shape=net_data["fc7"][0].shape)
        fc7b = tf.get_variable(name='fc7b', shape=net_data["fc7"][1].shape)
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

        # fc8 fc(1000, relu=False, name='fc8')
        fc8W = tf.get_variable(name='fc8W', shape=net_data["fc8"][0].shape)
        fc8b = tf.get_variable(name='fc8b', shape=net_data["fc8"][1].shape)
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

        return fc8

def alexnet(inputs, scope='alexnet'):
    net_data = load(open("/media/dataHD3/kpugdeet/PRE/bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    with tf.variable_scope(scope, 'alexnet', [inputs], reuse=None) as sc:

        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(inputs, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1 lrn(2, 2e-05, 0.75, name='norm1')
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool1 max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2 conv(5, 5, 256, 1, 1, group=2, name='conv2')
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2 lrn(2, 2e-05, 0.75, name='norm2')
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

        # maxpool2 max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3 conv(3, 3, 384, 1, 1, name='conv3')
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
        conv3 = tf.nn.relu(conv3_in)

        # conv4 conv(3, 3, 384, 1, 1, group=2, name='conv4')
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
        conv4 = tf.nn.relu(conv4_in)

        # conv5 conv(3, 3, 256, 1, 1, group=2, name='conv5')
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5 max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fc6 fc(4096, name='fc6')
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

        # fc7 fc(4096, name='fc7')
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

        # fc8 fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

        return fc8

