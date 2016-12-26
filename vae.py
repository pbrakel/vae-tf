import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


class MLP(object):

    def __init__(self, dims, activations):
        self.dims = dims
        self.activations = activations
        self.weights = []
        self.layers = []
        previous_dim = dims[0]
        for dim, activation in zip(dims[1:], activations):
            weights = weight_variable((previous_dim, dim))
            biases = bias_variable((dim,))
            self.layers.append((weights, biases, activation))
            previous_dim = dim

    def __call__(self, x):
        h = x
        for weights, biases, activation in self.layers:
            h = tf.matmul(h, weights) + biases
            if activation:
                h = activation(h)
        return h


class BinaryVAE(object):
    """A variational autoencoder for binary data.

    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = MLP([input_dim, hidden_dim, 2 * latent_dim],
                           [tf.nn.relu, None])
        self.decoder = MLP([latent_dim, hidden_dim, input_dim],
                           [tf.nn.relu, None])

    def neg_bound(self, x):
        """Returns the cost to minimize for training the VAE.
        
        This value is a sampling-based approximation of an upperbound of the
        negative log-likelihood of the sample(s) x.

        """
        mu_logsigma = self.encoder(x)

        mu = mu_logsigma[:, :self.latent_dim]
        logsigma = mu_logsigma[:, self.latent_dim:] - 1
        sigma = tf.exp(logsigma)

        eps = tf.random_normal(tf.shape(sigma))
        z_samp = eps * sigma + mu

        logits = self.decoder(z_samp)

        KL_sum = -tf.reduce_sum((1 + 2 * logsigma - mu**2 - sigma**2), 1) * .5
        CE = tf.nn.sigmoid_cross_entropy_with_logits(logits, x)
        CE_sum = tf.reduce_sum(CE, 1)
        return tf.reduce_mean(KL_sum + CE_sum)

    def sample(self, n, return_dist=False):
        """Sample z~p(z) and return either p(x | z) or x~p(x | z).

        Parameters
        ----------
        n : integer
            The number of samples to generate.
        return_dist : bool
            ``False`` by default. If ``True``, the method returns the parameters
            of p(x | z) instead of a sample from it.


        """
        z_samp = tf.random_normal((n, self.latent_dim))
        logits = self.decoder(z_samp)
        dist = tf.nn.sigmoid(logits)
        if return_dist:
            return dist
        else:
            flip = tf.random_uniform((n, self.latent_dim))
            x_samp = tf.sign(dist - flip) / 2 + .5
            return x_samp
