import tensorflow as tf
import numpy as np
from vae import BinaryVAE
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data


def tile_digits(rows, columns, data):
    if data.shape[0] != rows * columns:
        raise ValueError("The number of rows in 'data' should equal"
                         "'rows' * 'columns'")
    canvas = np.zeros((rows * 29 - 1, columns * 29 - 1))
    for row in xrange(rows):
        for column in xrange(columns):
            image = data[row * columns + column].reshape((28, 28))
            row_off = row * 29
            col_off = column * 29
            canvas[row_off:row_off + 28, col_off:col_off + 28] = image
    return canvas


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    vae = BinaryVAE(784, 500, 50)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    cost = vae.neg_bound(x)
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

    sess = tf.Session()
    init = tf.initialize_all_variables()

    sample = vae.sample(100, return_dist=True)

    sess.run(init)

    for i in range(100000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0]}, session=sess)
        if i % 1000 == 1:
            print sess.run(cost, feed_dict={x: batch[0]})
        if i % 5000 == 1:
            x_samp = sess.run(sample)
            images = np.uint8(tile_digits(10, 10, x_samp) * 255)
            pil_im = Image.fromarray(images).convert('RGB')
            pil_im.save("samples.png")
            print 'image saved to "samples.png"'


if __name__ == '__main__':
    main()
