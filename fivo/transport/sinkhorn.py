import tensorflow as tf
from fivo.transport.utils import squared_distances, softmin

MACHINE_PRECISION = 1e-10


def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, epsilon, threshold, max_iter):
    # initialisation
    a_y_init = softmin(epsilon, cost_yx, log_alpha)
    b_x_init = softmin(epsilon, cost_xy, log_beta)

    def apply_one(a_y, b_x):
        at_y = softmin(epsilon, cost_yx, log_alpha + b_x / epsilon)
        bt_x = softmin(epsilon, cost_xy, log_beta + a_y / epsilon)

        a_y_new = .5 * (a_y + at_y)
        b_x_new = .5 * (b_x + bt_x)

        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y))
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x))

        return a_y_new, b_x_new, tf.maximum(a_y_diff, b_x_diff)

    def stop_condition(i, u, v, update_size):
        n_iter_cond = i < max_iter - 1
        stable_cond = update_size > threshold
        precision_cond = tf.logical_and(tf.reduce_min(u) > MACHINE_PRECISION,
                                        tf.reduce_min(v) > MACHINE_PRECISION)
        return tf.reduce_all([n_iter_cond, stable_cond, precision_cond])

    def body(i, u, v, _update_size):
        new_u, new_v, new_update_size = apply_one(u, v)
        return i + 1, new_u, new_v, new_update_size

    n_iter = tf.constant(0)
    initial_update_size = tf.constant(2 * threshold)
    _total_iter, converged_a_y, converged_b_x, last_update_size = tf.while_loop(stop_condition, body,
                                                                                loop_vars=[n_iter,
                                                                                           a_y_init,
                                                                                           b_x_init,
                                                                                           initial_update_size])

    # We do a last extrapolation for the gradient - leverages fixed point + implicit function theorem
    a_y = softmin(epsilon, cost_yx, log_alpha + tf.stop_gradient(converged_b_x) / epsilon)
    b_x = softmin(epsilon, cost_xy, log_beta + tf.stop_gradient(converged_a_y) / epsilon)

    return a_y, b_x


def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, threshold, max_iter=100):
    cost_xy = 0.5 * squared_distances(x, tf.stop_gradient(y))
    cost_yx = 0.5 * squared_distances(y, tf.stop_gradient(x))
    a_y, b_x = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, epsilon, threshold, max_iter)
    return a_y, b_x


if __name__ == '__main__':
    import numpy as np
    import os
    import torch
    from geomloss.sinkhorn_samples import sinkhorn_tensorized
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto(device_count={'GPU': 0})

    np.random.seed(0)

    N = 100

    w = np.random.uniform(0.1, 2., (2, N)).astype(np.float32)
    w /= w.sum()
    log_w = np.log(w)
    x = np.random.normal(0., 1., (2, N, 3)).astype(np.float32)
    log_w_uniform = -np.log(N) * np.ones_like(w)

    log_w_tf = tf.constant(log_w)
    log_w_uniform_tf = tf.constant(log_w_uniform)
    tf_x = tf.constant(x)
    potentials = sinkhorn_potentials(log_w_tf, tf_x, log_w_uniform_tf, tf_x, 1., 1e-1)
    potential_grad = tf.gradients(tf.reduce_sum(potentials[0]), tf_x)

    with tf.Session(config=config) as sess:
        tic = time.time()
        potentials_val = sess.run(potentials)
        potential_grad_val = sess.run(potential_grad)
    print(time.time() - tic)

    print(potentials_val[0].shape)

    torch_w = torch.tensor(w, requires_grad=True).cpu()
    torch_x = torch.tensor(x, requires_grad=True).cpu()
    tic = time.time()
    geomloss_potentials = sinkhorn_tensorized(torch_w,
                                              torch_x,
                                              torch.tensor(log_w_uniform).exp(),
                                              torch_x,
                                              blur=np.sqrt(1.),
                                              debias=False,
                                              potentials=True, scaling=0.995)
    print()
    print(torch.autograd.grad(geomloss_potentials[1].sum(), torch_x)[0].numpy())
    print(time.time() - tic)
