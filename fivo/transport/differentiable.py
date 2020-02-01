import tensorflow as tf
from utils import squared_distances
from sinkhorn import sinkhorn_potentials
import math


def transport_from_potentials(x, f, g, eps, w, n):
    """
    To get the transported particles from the sinkhorn iterates
    :param x: tf.Tensor[N, D]
        Input: the state variable
    :param f: tf.Tensor[N]
        Potential, output of the sinkhorn iterates
    :param g: tf.Tensor[N]
        Potential, output of the sinkhorn iterates
    :param eps: float
    :param w: torch.Tensor[N]
    :param n: int
    :return: tf.Tensor[N, D], tf.Tensor[N]
    """

    cost_matrix = squared_distances(x, x) / 2

    fg = tf.einsum('ij,ik->ijk', f, g)
    denominator = eps

    # Clip minimal contributions to prevent nans in gradient
    # i.e. weight can't be smaller than 1/n**2
    temp = tf.clip_by_value(fg - cost_matrix,
                            -2 * denominator * math.log(n),
                            float('inf'))

    temp /= denominator
    transport_matrix = tf.math.exp(temp) * tf.expand_dims(w, 2)
    transport_matrix /= tf.reduce_sum(transport_matrix, axis=2, keepdims=True)
    uniform_log_weight = -math.log(n) * tf.ones_like(w)
    return tf.einsum('ijk,ikl->ijl', transport_matrix, x), uniform_log_weight


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
    alpha, beta = sinkhorn_potentials(uniform_log_weight, x, logw, x, eps, threshold, max_iter)
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
    alpha, beta = solve_for_state(x, logw, eps, threshold, max_iter, n)
    x_tilde, w_tilde = transport_from_potentials(x, alpha, beta, eps, tf.math.exp(logw), n)
    return x_tilde, w_tilde


def get_transport_fun(eps, threshold, max_iter=100):
    def fun(log_weights, states, num_particles, _batch_size, _random_seed=None):
        return transport(states, tf.transpose(log_weights), eps, threshold, num_particles, max_iter)

    return fun


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
    particles = transport(tf_x, log_w_tf, 1e-1, 1e-2, 100, N)

    with tf.Session(config=config) as sess:
        tic = time.time()
        particles_val, log_uniform_weights = sess.run(particles)
        print(time.time() - tic)

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].hist(particles_val[0, :, 0], weights=np.exp(log_w_uniform)[0, :], label='transported', color='orange')
    axes[1].hist(x[0, :, 0], weights=w[0, :], label='original', color='blue')
    fig.legend()
    plt.show()
    print(particles_val.mean(axis=1))
    print(np.mean(x * w[:, :, None], axis=1))
