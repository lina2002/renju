import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from parse import my_get_data
from tqdm import tqdm
import numpy as np


init_scale = 0.05
batch_size = 32
num_of_epochs = 10
is_training = tf.placeholder_with_default(False, shape=(), name='is_training')


def var(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-init_scale, maxval=init_scale))


N = 128
weights_list = [var([7, 7, N, N]),
           var([7, 7, N, N]),
           var([15*15*128, 3072]),
           var([3072, 225])]


def WeightsGenerator():
    for w in weights_list:
        yield w


weights = WeightsGenerator()


bn_params = {
    'is_training': is_training,
    'decay': 1,
    'updates_collections': None
}


def residual(x):
    to_add = x
    x = tf.nn.conv2d(x, next(weights), strides=[1, 1, 1, 1], padding="SAME")
    x = tf.nn.relu(x)
    x = batch_norm(x, **bn_params)
    x = tf.nn.conv2d(x, next(weights), strides=[1, 1, 1, 1], padding="SAME")
    x = tf.add_n([x, to_add])
    return x


x = tf.placeholder(tf.float32, [None, 15, 15, 5])
y = tf.placeholder(tf.float32, [None, 225])

z = tf.pad(x, tf.constant([[0, 0], [0, 0], [0, 0], [0, N-5]]))  # paddings shape 4x2
z = residual(z)
z = tf.nn.relu(z)
z = batch_norm(z, **bn_params)
z = tf.reshape(z, (-1, 15*15*N))
z = tf.matmul(z, next(weights))
z = tf.nn.relu(z)
z = tf.matmul(z, next(weights))
z = batch_norm(z, **bn_params)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    boards, moves = my_get_data('./data/train.xml')
    for epoch in range(num_of_epochs):
        print("epoch number: " + str(epoch + 1))
        permuted_indices = np.random.permutation(boards.shape[0])
        for i in tqdm(range(0, boards.shape[0], batch_size)):
            selected_data_points = np.take(permuted_indices, range(i, i+batch_size), mode='wrap')
            sess.run(optimizer, {x: boards[selected_data_points], y: moves[selected_data_points], is_training: True})
