import tensorflow as tf


class MultinomialResampling(tf.Module):
    def __init__(self, num_particles, batch_size, name=None):
        super(MultinomialResampling, self).__init__(name=name)
        self.num_particles = num_particles
        self.batch_size = batch_size

    def __call__(self, inp):


def multinomial_resampling(log_weights, states, num_particles, batch_size,
                           random_seed=None):
  """Resample states with multinomial resampling.

  Args:
    log_weights: A [num_particles, batch_size] Tensor representing a batch
      of batch_size logits for num_particles-ary Categorical distribution.
    states: A nested list of [batch_size*num_particles, data_size] Tensors that
      will be resampled from the groups of every num_particles-th row.
    num_particles: The number of particles/samples.
    batch_size: The batch size.
    random_seed: The random seed to pass to the resampling operations in
      the particle filter. Mainly useful for testing.

  Returns:
    resampled_states: A nested list of [batch_size*num_particles, data_size]
      Tensors resampled via multinomial sampling.
  """
  # Calculate the ancestor indices via resampling. Because we maintain the
  # log unnormalized weights, we pass the weights in as logits, allowing
  # the distribution object to apply a softmax and normalize them.
  resampling_parameters = tf.transpose(log_weights, perm=[1, 0])
  resampling_dist = tf.contrib.distributions.Categorical(
      logits=resampling_parameters)
  ancestors = tf.stop_gradient(
      resampling_dist.sample(sample_shape=num_particles, seed=random_seed))

  # Because the batch is flattened, we must modify ancestor_inds to index the
  # proper samples. The particles in the ith filter are distributed every
  # batch_size rows in the batch, and offset i rows from the top. So, to
  # correct the indices we multiply by the batch_size and add the proper offset.
  # Crucially, when ancestor_inds is flattened the layout of the batch is
  # maintained.
  offset = tf.expand_dims(tf.range(batch_size), 0)
  ancestor_inds = tf.reshape(ancestors * batch_size + offset, [-1])

  resampled_states = nested.gather_tensors(states, ancestor_inds)
  return resampled_states