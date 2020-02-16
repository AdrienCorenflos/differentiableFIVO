
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




def transport_from_potentials(x, f, g, eps, logw, n, ):
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
    # op_transport_matrix = tf.print('transport_matrix: ', transport_matrix)
    #
    # with tf.control_dependencies([op_w]):#, op_fg, op_temp, op_exp_temp, op_transport_matrix]):
    res = tf.einsum('ijk,ikl->ijl', transport_matrix, x,)
    uniform_log_weight = -math.log(n) * tf.ones_like(logw)
    return res, uniform_log_weight


def solve_for_state(x, logw, eps, threshold, max_iter, n):
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
    alpha, beta, _, _ = sinkhorn_potentials(uniform_log_weight, x, logw, x, eps, threshold, max_iter)
    return alpha, beta


def transport(x, logw, eps, threshold, n, max_iter):
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
    alpha, beta = solve_for_state(x, logw, eps, threshold, max_iter, n)
    #print_op1 = tf.print('alpha', alpha.shape)
    #print_op2 = tf.print('beta', beta.shape)
    #with tf.control_dependencies([print_op1,print_op2]):
    x_tilde, w_tilde = transport_from_potentials(x, alpha, beta, eps, logw, n)
    return x_tilde, w_tilde


def transport_helper(x, log_weights, num_particles, batch_size, eps, threshold, max_iter):
    if len(x.get_shape().as_list()) > 1:
        data_dim = x.get_shape().as_list()[-1]
    else:
        data_dim = 1
    #print_op1 = tf.print("pre reshape x: ", x.shape)
    #with tf.control_dependencies([print_op1]):
    reshaped_state = tf.reshape(x, [batch_size, num_particles, data_dim])

    x_tilde, _ = transport(reshaped_state, tf.transpose(log_weights), eps, threshold, num_particles, max_iter)

    x_tilde = tf.reshape(x_tilde, [batch_size * num_particles, data_dim])

    return x_tilde


def vrnn_transport_resamp(eps, threshold, max_iter=100):

    def func(log_weights, states, num_particles, batch_size, **_kwargs):
        data_dim = states.latent_encoded.get_shape().as_list()[-1]

        # get tensors from state
        rnn_state = states.rnn_state
        rnn_out = states.rnn_out # this does not get used, but need a placeholder
        latent_encoded = states.latent_encoded

        # concat
        state_tensor = tf.concat([rnn_state, latent_encoded], axis=1)

        # run transport resampling
        new_state_tensor = transport_helper(state_tensor, log_weights, num_particles, batch_size, eps, threshold, max_iter)
        new_rnn_state, new_latent_encoded = tf.split(new_state_tensor, num_or_size_splits = [data_dim*2, data_dim], axis=1)


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
