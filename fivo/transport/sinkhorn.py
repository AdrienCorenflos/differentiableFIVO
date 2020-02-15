import tensorflow as tf
# from fivo.transport.utils import squared_distances, softmin
from .utils import squared_distances, softmin

MACHINE_PRECISION = 1e-10


def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, threshold, max_iter):
    # initialisation
    a_y_init = softmin(epsilon, cost_yx, log_alpha)
    b_x_init = softmin(epsilon, cost_xy, log_beta)
    a_x_init = softmin(epsilon, cost_xx, log_alpha)
    b_y_init = softmin(epsilon, cost_yy, log_beta)

    def apply_one(a_y, b_x, a_x, b_y):
        at_y = softmin(epsilon, cost_yx, log_alpha + b_x / epsilon)
        bt_x = softmin(epsilon, cost_xy, log_beta + a_y / epsilon)
        at_x = softmin(epsilon, cost_xx, log_alpha + a_x / epsilon)
        bt_y = softmin(epsilon, cost_yy, log_beta + b_y / epsilon)

        a_y_new = .5 * (a_y + at_y)
        b_x_new = .5 * (b_x + bt_x)
        a_x_new = .5 * (a_x + at_x)
        b_y_new = .5 * (b_y + bt_y)

        a_y_diff = tf.reduce_max(tf.abs(a_y_new - a_y))
        b_x_diff = tf.reduce_max(tf.abs(b_x_new - b_x))

        return a_y_new, b_x_new, a_x_new, b_y_new, tf.maximum(a_y_diff, b_x_diff)

    def stop_condition(i, a_y, b_x, _a_x, _b_y, update_size):
        n_iter_cond = i < max_iter - 1
        stable_cond = update_size > threshold
        precision_cond = tf.logical_and(tf.reduce_min(a_y) > MACHINE_PRECISION,
                                        tf.reduce_min(b_x) > MACHINE_PRECISION)
        return tf.reduce_all([n_iter_cond, stable_cond, precision_cond])

    def body(i, a_y, b_x, a_x, b_y, _update_size):
        a_y_new, b_x_new, a_x_new, b_y_new, new_update_size = apply_one(a_y, b_x, a_x, b_y)
        return i + 1, a_y_new, b_x_new, a_x_new, b_y_new, new_update_size

    n_iter = tf.constant(0)
    initial_update_size = tf.constant(2 * threshold)
    _total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, last_update_size = tf.while_loop(
        stop_condition, body,
        loop_vars=[n_iter,
                   a_y_init,
                   b_x_init,
                   a_x_init,
                   b_y_init,
                   initial_update_size])

    # We do a last extrapolation for the gradient - leverages fixed point + implicit function theorem
    a_y_final = softmin(epsilon, cost_yx, log_alpha + tf.stop_gradient(converged_b_x) / epsilon)
    b_x_final = softmin(epsilon, cost_xy, log_beta + tf.stop_gradient(converged_a_y) / epsilon)

    a_x_final = softmin(epsilon, cost_xx, log_alpha + tf.stop_gradient(converged_a_x) / epsilon)
    b_y_final = softmin(epsilon, cost_yy, log_beta + tf.stop_gradient(converged_b_y) / epsilon)

    return a_y_final, b_x_final, a_x_final, b_y_final


def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, threshold, max_iter=100):
    cost_xy = 0.5 * squared_distances(x, tf.stop_gradient(y))
    cost_yx = 0.5 * squared_distances(y, tf.stop_gradient(x))
    cost_xx = 0.5 * squared_distances(x, tf.stop_gradient(x))
    cost_yy = 0.5 * squared_distances(y, tf.stop_gradient(y))
    a_y, b_x, a_x, b_y = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, threshold,
                                       max_iter)
    return a_y, b_x, a_x, b_y


if __name__ == '__main__':
    import numpy as np
    import os
    # import torch
    # from geomloss.sinkhorn_samples import sinkhorn_tensorized
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
