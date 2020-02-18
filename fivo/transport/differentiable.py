
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from fivo.transport.utils import squared_distances
from fivo.transport.sinkhorn import sinkhorn_potentials
from fivo.nested_utils import map_nested
import math
from tensorflow.python.util import nest
from fivo.models.vrnn import TrainableVRNNState

import sys




def transport_from_potentials(x, f, g, eps, logw, n, other_things_to_transport = None):
    """
    To get the transported particles from the sinkhorn iterates
    :param x: tf.Tensor[N, D]
        Input: the state variable
    :param f: tf.Tensor[N]
        Potential, output of the sinkhorn iterates
    :param g: tf.Tensor[N]
        Potential, output of the sinkhorn iterates
    :param eps: float
    :param logw: torch.Tensor[N]
    :param n: int
    :return: tf.Tensor[N, D], tf.Tensor[N]
    """

    cost_matrix = squared_distances(x, x) / 2

    fg = tf.einsum('ij,ik->ijk', f, g)
    denominator = eps

    # Clip minimal contributions to prevent nans in gradient
    # i.e. weight can't be smaller than 1/n**3
    temp = tf.clip_by_value(fg - cost_matrix,
                            tf.minimum(-3 * denominator * math.log(n), -denominator*math.log(6)),
                            float('inf'))

    temp /= denominator
    transport_matrix = tf.math.exp(temp + tf.expand_dims(logw, 2))
    transport_matrix /= tf.reduce_sum(transport_matrix, axis=2, keepdims=True)
    # op_fg = tf.print('fg:', fg)
    # op_w = tf.print('w:', tf.reduce_logsumexp(logw, 1))
    # op_temp = tf.print('temp:', temp)
    # op_exp_temp = tf.print('temp exp:', tf.math.exp(temp))
    #op_transport_matrix = tf.print('transport_matrix: ', transport_matrix.shape)
    #with tf.control_dependencies([op_transport_matrix]):#, op_fg, op_temp, op_exp_temp, op_transport_matrix]):
    res = tf.einsum('ijk,ikl->ijl', transport_matrix, x)
    uniform_log_weight = -math.log(n) * tf.ones_like(logw)

    if other_things_to_transport is not None:
        other_things_transported = tf.einsum('ijk,ikl->ijl', transport_matrix, other_things_to_transport)
    else:
        other_things_transported = None
    return res, uniform_log_weight, other_things_transported


def solve_for_state(x, logw, eps, threshold, max_iter, n, batch_size):
    """
    :param x: tf.Tensor[N, D]
        The input
    :param logw: tf.Tensor[N]
        The degenerate logweights
    :param eps: float
    :param threshold: float
    :param max_iter: int
    :param n: int
    :return: torch.Tensor[N], torch.Tensor[N]
        the potentials
    """
    uniform_log_weight = -math.log(n) * tf.ones_like(logw)
    alpha, beta, _, _ = sinkhorn_potentials(uniform_log_weight, x, logw, x, eps, threshold, batch_size, max_iter)
    return alpha, beta


def transport(x, logw, eps, threshold, n, max_iter, batch_size, other_things_to_transport=None):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: tf.Tensor[N, D]
        The input
    :param logw: tf.Tensor[N]
        The degenerate logweights
    :param eps: float
    :param threshold: float
    :param n: int
    :param max_iter: int
    """
    # normalized_logw_print_op = tf.print( "normalized log: ", tf.reduce_min(logw))
    # with tf.control_dependencies([normalized_logw_print_op]):
    alpha, beta = solve_for_state(x, logw, eps, threshold, max_iter, n, batch_size)
    #print_op1 = tf.print('alpha', alpha.shape)
    #print_op2 = tf.print('beta', beta.shape)
    #with tf.control_dependencies([print_op1,print_op2]):
    x_tilde, w_tilde, other_things_transported = transport_from_potentials(x, alpha, beta, eps, logw, n, other_things_to_transport)
    return x_tilde, w_tilde, other_things_transported


def transport_helper(x, log_weights, num_particles, batch_size, eps, threshold, max_iter, other_things_to_transport=None):
    if len(x.get_shape().as_list()) > 1:
        data_dim = x.get_shape().as_list()[-1]
    else:
        data_dim = 1
    #print_op1 = tf.print("pre reshape x: ", x.shape)
    #with tf.control_dependencies([print_op1]):
    reshaped_state = tf.reshape(x, [batch_size, num_particles, data_dim])
    if other_things_to_transport is not None:
        other_data_dim = other_things_to_transport.get_shape().as_list()[-1]
        other_things_to_transport = tf.reshape(other_things_to_transport, [batch_size, num_particles, other_data_dim])

    #print_op = tf.print('reshaped_state', reshaped_state.shape)
    #with tf.control_dependencies([print_op]):
    x_tilde, _ , other_things_transported= transport(reshaped_state, tf.transpose(log_weights), eps, threshold, num_particles, max_iter, batch_size, other_things_to_transport)

    #print_op = tf.print('x_tilde',x_tilde.shape)
    #with tf.control_dependencies([print_op]):
    x_tilde = tf.reshape(x_tilde, [batch_size * num_particles, data_dim])
    if other_things_to_transport is not None:
        other_things_transported = tf.reshape(other_things_to_transport, [batch_size * num_particles, other_data_dim])

    return x_tilde, other_things_transported


def vrnn_transport_resamp(eps, threshold, max_iter=100, transport_rnn_state = False):

    def func(log_weights, states, num_particles, batch_size, **_kwargs):
        data_dim = states.latent_encoded.get_shape().as_list()[-1]

        # get tensors from state
        rnn_state = states.rnn_state
        rnn_out = states.rnn_out # this does not get used, but need a placeholder
        latent_encoded = states.latent_encoded


        ot_all = transport_rnn_state
        # run transport resampling
        if ot_all: # transport all state
            # concat
            state_tensor = tf.concat([rnn_state, latent_encoded], axis=1)

            new_state_tensor, _ = transport_helper(state_tensor,
                                                                 log_weights,
                                                                 num_particles,
                                                                 batch_size,
                                                                 eps,
                                                                 threshold,
                                                                 max_iter)

            new_rnn_state, new_latent_encoded = tf.split(new_state_tensor, num_or_size_splits = [data_dim*2, data_dim], axis=1)

        if not transport_rnn_state:
            #print_op = tf.print('latent_encoded', latent_encoded.shape)
            #with tf.control_dependencies([print_op]):
            new_latent_encoded, new_rnn_state = transport_helper(latent_encoded,
                                                                 log_weights,
                                                                 num_particles,
                                                                 batch_size,
                                                                 eps,
                                                                 threshold,
                                                                 max_iter,
                                                                 other_things_to_transport=rnn_state)

        new_state = TrainableVRNNState(rnn_state=new_rnn_state,
                                       rnn_out=rnn_out,
                                       latent_encoded=new_latent_encoded)

        return new_state
    return func


def get_transport_fun(eps, threshold, max_iter=100, model_tag = None):
    print(model_tag)
    if model_tag == "vrnn":
        func = vrnn_transport_resamp(eps, threshold, max_iter=100)
    else:
        def func(log_weights, states, num_particles, batch_size, **_kwargs):
            new_states = transport_helper(states, log_weights, num_particles, batch_size, eps, threshold, max_iter)
            new_states = tf.reshape(new_states, [batch_size * num_particles])
            return new_states

    return func


if __name__ == '__main__':
    import numpy as np
    import os
    import time
    import matplotlib.pyplot as plt

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # tf.enable_eager_execution()
    config = tf.ConfigProto(device_count={'GPU': 0})

    np.random.seed(0)

    N = 250

    w = np.random.uniform(0.1, 2., (2, N)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    log_w = np.log(w)
    x = np.random.normal(0., 1., (2, N, 3)).astype(np.float32)
    log_w_uniform = -np.log(N) * np.ones_like(w)

    log_w_tf = tf.constant(log_w)
    log_w_uniform_tf = tf.constant(log_w_uniform)
    tf_x = tf.constant(x)
    particles = transport(tf_x, log_w_tf, 5e-1, 1e-2, 100, N)
    d_particles = tf.gradients(particles[0], tf_x)
    with tf.Session(config=config) as sess:
        tic = time.time()
        particles_val, log_uniform_weights = sess.run(particles)
        print(sess.run(d_particles))
        print(time.time() - tic)

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].hist(particles_val[0, :, 0], weights=np.exp(log_w_uniform)[0, :], label='transported', color='orange')
    axes[1].hist(x[0, :, 0], weights=w[0, :], label='original', color='blue')
    fig.legend()
    plt.show()
    print(particles_val.mean(axis=1))
    print(np.mean(x * w[:, :, None], axis=1))
