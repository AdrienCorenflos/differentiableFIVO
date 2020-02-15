import tensorflow as tf
# from fivo.transport.sinkhorn import sinkhorn_potentials
from sinkhorn import sinkhorn_potentials
import math


def _cost(a_y, b_x, a_x, b_y, logw, n):
    uniform_weight = tf.zeros_like(logw) + 1 / n
    x_tensor = b_x - a_x
    x_cost = tf.reduce_sum(tf.math.sign(x_tensor) * tf.exp(tf.log(x_tensor) + logw), -1, keepdims=False)
    return x_cost + tf.einsum('ij,ij->i', uniform_weight, a_y - b_y)


def _learn(x, logw, init_x, eps, threshold, n, max_iter_inner, max_iter_outer, learning_rate):
    uniform_log_weight = tf.zeros_like(logw) - math.log(n)

    def body(i, z):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(z)
            a_y, b_x, a_x, b_y = sinkhorn_potentials(logw, x, uniform_log_weight, z, eps, threshold, max_iter_inner)
            batch_cost = _cost(a_y, b_x, a_x, b_y, logw, uniform_log_weight)
            costs = tf.unstack(batch_cost)
        grads = [tape.gradient(cost, z) for cost in costs]

        update = learning_rate * tf.math.add_n(grads)

        return i + 1, z - update

    def stop_condition(i, _z):
        n_iter_cond = i < max_iter_outer
        return n_iter_cond

    n_iter = tf.constant(0)
    _total_iter, final_z = tf.while_loop(
        stop_condition, body,
        loop_vars=[n_iter,
                   tf.stop_gradient(init_x)])
    return final_z


def learn(x, logw, n, epsilon, inner_threshold, max_iter_inner, max_iter_outer, init_fun,
          learning_rate=1.):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: tf.Tensor[N, D]
        The input
    :param logw: tf.Tensor[N]
        The degenerate log weights
    :param n: int
        number of particles
    :param epsilon: float
        Blur parameter used in loss
    :param inner_threshold: float
        argument for sinkhorn
    :param max_iter_inner: int
        argument for sinkhorn
    :param max_iter_outer: int
        number of steps for optimisation
    :param init_fun: callable
        a smart starting point
    """

    init_x = init_fun(x, logw, n)
    learnt_x = _learn(x, logw, init_x, epsilon, inner_threshold, n, max_iter_inner, max_iter_outer, learning_rate)
    uniform_log_weight = tf.tile(tf.constant([-math.log(n)]), tf.constant([n]))

    return learnt_x, uniform_log_weight


def get_learn_fun(eps, inner_threshold, inner_max_iter=20, outer_max_iter=10, learning_rate=0.1,
                  initialisation_style=None):
    if initialisation_style is None:
        init_fun = lambda x, logw, n: tf.identity(x)

    def fun(log_weights, x, num_particles, _batch_size, **_kwargs):
        learnt_x, uniform_log_weight = learn(log_weights, x, num_particles, eps, inner_threshold, inner_max_iter,
                                             outer_max_iter, init_fun, learning_rate)
        return learnt_x, uniform_log_weight

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

    N = 200


    x = np.random.normal(0., 1., (3, N, 2)).astype(np.float32)
    w = np.ones((3, N), dtype=np.float32)
    w[x[:, :, 0] > 0] = 1e-6
    w /= w.sum(axis=1, keepdims=True)
    log_w = np.log(w)
    log_w_uniform = -np.log(N) * np.ones_like(w)


    log_w_tf = tf.constant(log_w)
    log_w_uniform_tf = tf.constant(log_w_uniform)
    tf_x = tf.constant(x)
    init_fun = lambda x, logw, n: tf.identity(x)
    particles = learn(tf_x, log_w_tf, N, 0.1, 1e-6, 30, 50, init_fun, 0.5)
    d_particles = tf.gradients(particles[0], tf_x)
    print(d_particles)
    with tf.Session(config=config) as sess:
        tic = time.time()
        particles_val, log_uniform_weights = sess.run(particles)
        # print(sess.run(d_particles))
        print(time.time() - tic)

    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
    axes[0].hist(particles_val[0, :, 0], weights=np.exp(log_w_uniform)[0, :], label='learned', color='orange', bins=20)
    axes[1].hist(x[0, :, 0], weights=np.exp(log_w_uniform)[0, :], label='naive', color='green', bins=20)
    axes[2].hist(x[0, :, 0], weights=w[0, :], label='original', color='blue', bins=20)
    fig.legend()
    plt.show()
    print(particles_val.mean(axis=1))
    print(np.mean(x * w[:, :, None], axis=1))
