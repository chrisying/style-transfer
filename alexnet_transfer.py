################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image


im_content = (imread("laska.png")[:,:,:3]).astype(float32)
im_content = im_content - mean(im_content)

im_style = (imread("starry.png")[:,:,:3]).astype(float32)
im_style = im_style - mean(im_style)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)

def gram_matrix(conv_maps):
    map_shape = conv_maps.get_shape().as_list()
    z = tf.reshape(conv_maps, [map_shape[1] * map_shape[2], map_shape[3]])
    return tf.matmul(tf.transpose(z), z), map_shape[1] * map_shape[2]

# TODO: use variable scopes so that weights are initialized once
def conv5_and_grams(x_in):
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x_in, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    g1, m1 = gram_matrix(conv1)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    g2, m2 = gram_matrix(conv2)

    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    g3, m3 = gram_matrix(conv3)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    g4, m4 = gram_matrix(conv4)

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)
    g5, m5 = gram_matrix(conv5)

    #return conv5, g1, g2, g3, g4, g5, [m1,m2,m3,m4,m5]
    return conv5, g4, m4

#conv5, g1, g2, g3, g4, g5, ms = conv5_and_grams(x)
conv5, g4, m4 = conv5_and_grams(x)
#ns = map(lambda g:g.get_shape()[0], [g1, g2, g3, g4, g5])
n4 = g4.get_shape().as_list()[0]

## Reconstruction
content = tf.placeholder(tf.float32, conv5.get_shape()[1:])
style = tf.placeholder(tf.float32, g4.get_shape())
#recon = tf.Variable(tf.random_normal((1,) + xdim, stddev=1))
im_a = np.reshape(im_content, (1,)+im_content.shape)
recon = tf.Variable(im_a)

#r_conv5, r_g1, r_g2, r_g3, r_g4, r_g5, _ = conv5_and_grams(recon)
r_conv5, r_g4, _ = conv5_and_grams(recon)
r_conv5_reshape = tf.reshape(r_conv5, r_conv5.get_shape()[1:])
loss_content = 0.5 * tf.reduce_sum(tf.square(tf.subtract(content, r_conv5_reshape)))
#w1 = 1.0; w2 = 1.0; w3 = 1.0; w4 = 1.0; w5 = 1.0
#loss_style = 0.25 * (w1/(ns[0]**2 * ms[0]**2) * tf.reduce_sum(tf.square(tf.subtract(g1, r_g1)))
#                   + w2/(ns[1]**2 * ms[1]**2) * tf.reduce_sum(tf.square(tf.subtract(g2, r_g2)))
#                   + w3/(ns[2]**2 * ms[2]**2) * tf.reduce_sum(tf.square(tf.subtract(g3, r_g3)))
#                   + w4/(ns[3]**2 * ms[3]**2) * tf.reduce_sum(tf.square(tf.subtract(g4, r_g4)))
#                   + w5/(ns[4]**2 * ms[4]**2) * tf.reduce_sum(tf.square(tf.subtract(g5, r_g5))))
loss_style = 0.5 / (n4 ** 2 * m4 ** 2) * tf.reduce_sum(tf.square(tf.subtract(style, r_g4)))
alpha = 1.0; beta = 10000.0
# TODO: maybe add Lagrange multiplier to discourage extreme values
loss = alpha * loss_content + beta * loss_style
#opt_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=[recon])
opt_op = tf.train.AdamOptimizer().minimize(loss, var_list=[recon])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

t = time.time()

content_feature = sess.run(conv5, feed_dict={x:[im_content]})
style_matrix = sess.run(g4, feed_dict={x:[im_style]})
for i in xrange(100000):
    l, _ = sess.run([loss, opt_op], feed_dict={content: content_feature[0, :, :, :], style: style_matrix})
    print 'Iter: %d, Loss: %f' % (i, l)
r_im = sess.run(recon, feed_dict={content: content_feature[0, :, :, :]})[0, :, :, :]
save('raw_image.npy', r_im)
min_im = np.min(r_im)
max_im = np.max(r_im)
range_im = max_im - min_im
r_im_scaled = (r_im - min_im) * 255.0 / range_im

# TODO: this scaling is bad
imsave('reconstruction.png', r_im_scaled)

print time.time()-t
