import tensorflow as tf


def softmin(epsilon, cost_matrix, f):
    batch_size = cost_matrix.get_shape().as_list()[-1]
    f_ = tf.reshape(f, [batch_size, 1, -1])
    temp_val = f_ - cost_matrix / epsilon
    log_sum_exp = tf.reduce_logsumexp(temp_val, axis=2)
    return -epsilon * log_sum_exp


def squared_distances(x: tf.Tensor, y: tf.Tensor):
    ndim = tf.keras.backend.ndim(x)
    if ndim == 2:
        # x.shape = [N, D]
        # y.shape = [M, D]
        xx = tf.reduce_sum(x * x, axis=1, keepdims=True)
        xy = x @ tf.transpose(y)
        yy = tf.expand_dims(tf.reduce_sum(y * y, axis=1), 0)
    elif ndim == 3:
        xx = tf.reduce_sum(x * x, axis=-1, keepdims=True)
        xy = tf.einsum('bnd,bmd->bnm', x, y)
        yy = tf.expand_dims(tf.reduce_sum(y * y, axis=-1), 1)
    else:
        raise ValueError('2 or 3 dimensions expected')
    return tf.clip_by_value(xx - 2 * xy + yy, 0., float('inf'))


if __name__ == '__main__':
    import numpy as np
    import torch
    from geomloss.sinkhorn_samples import softmin_tensorized

    config = tf.ConfigProto(device_count={'GPU': 0})

    np.random.seed(53)
    np_x = np.random.normal(0., 1., (3, 5)).astype(np.float32)
    np_y = np.log(np.random.uniform(0., 1., (3,)).astype(np.float32))

    x = tf.constant(np_x)
    y = tf.constant(np_y)

    torch_x = torch.tensor(np_x, requires_grad=True)
    torch_y = torch.tensor(np_y, requires_grad=True)
    torch_softmin = softmin_tensorized(0.01, torch_x, torch_y)
    print(torch_softmin.detach().numpy())
    with tf.Session(config=config) as sess:
        val = softmin(0.01, x, y)
        tf_val = sess.run(val)
        # tf_grad_x, tf_grad_y = sess.run(tf.gradients(tf.reduce_sum(val), [x, y]))

    torch_grad_x, torch_grad_y = torch.autograd.grad(torch_softmin.sum(), [torch_x, torch_y])

    print(tf_val)
