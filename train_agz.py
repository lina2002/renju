from parse import my_get_data

import functools
import tensorflow as tf
from tqdm import tqdm
import numpy as np


np.set_printoptions(threshold=np.nan)


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions == labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all


init_scale = 0.05
batch_size = 32
num_of_epochs = 100
is_training = tf.placeholder_with_default(False, shape=(), name='is_training')


def var(shape):
    # return tf.Variable(tf.random_uniform(shape, minval=-init_scale, maxval=init_scale))
    return tf.Variable(tf.glorot_uniform_initializer()(shape))


N = 128

weights_list = [var([1, 1, N, 1])]


def WeightsGenerator():
    for w in weights_list:
        yield w


weights = WeightsGenerator()


x = tf.placeholder(tf.float32, [None, 15, 15, 5])
y = tf.placeholder(tf.float32, [None, 225])


my_batchn = functools.partial(
    tf.layers.batch_normalization,
    axis=-1,
    momentum=.95,
    epsilon=1e-5,
    center=True,
    scale=True,
    fused=True,
    training=is_training)

my_conv2d = functools.partial(
    tf.layers.conv2d,
    filters=N,
    kernel_size=3,
    padding="same",
    data_format="channels_last",
    use_bias=False)


def my_res_layer(inputs):
    int_layer1 = my_batchn(my_conv2d(inputs))
    initial_output = tf.nn.relu(int_layer1)
    int_layer2 = my_batchn(my_conv2d(initial_output))
    output = tf.nn.relu(inputs + int_layer2)
    return output


z = tf.nn.relu(my_batchn(my_conv2d(x)))

for _ in range(3):
    z = my_res_layer(z)

z = tf.nn.conv2d(z, next(weights), strides=[1, 1, 1, 1], padding="SAME")
z = tf.reshape(z, (-1, 225))

y_pred = tf.nn.softmax(z)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(cross_entropy)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def identity(x):
    return x


def compose2(f, g):
    return lambda x: f(g(x))


def compose3(f, g, h):
    return lambda x: f(g(h(x)))


def compose4(f, g, h, j):
    return lambda x: f(g(h(j(x))))


transformations = [identity, np.rot90, compose2(np.rot90, np.rot90), compose3(np.rot90, np.rot90, np.rot90),
                   np.fliplr, compose2(np.rot90, np.fliplr), compose3(np.rot90, np.rot90, np.fliplr),
                   compose4(np.rot90, np.rot90, np.rot90, np.fliplr)]

my_rot = functools.partial(np.rot90, axes=(1, 2))


def predict(bs):
    inputs = [bs, my_rot(bs)]
    predictions = [y_pred.eval({x: inp}) for inp in inputs]

    predictions[1] = np.reshape(predictions[1], (100, 15, 15))
    predictions[1] = my_rot(my_rot(my_rot(predictions[1])))
    predictions[1] = np.reshape(predictions[1], (100, 225))

    max_0 = np.max(predictions[0], axis=1)
    max_1 = np.max(predictions[1], axis=1)
    take_first = max_0 >= max_1
    results = [a_0 if t else a_1 for t, a_0, a_1 in zip(take_first, np.argmax(predictions[0], axis=1), np.argmax(predictions[1], axis=1))]
    return results


def get_predictions(bs):
    preditions = []
    for i in tqdm(range(0, bs.shape[0], 100)):
        preditions = np.append(preditions, predict(bs[i:(i+100)]))
    return preditions


if __name__ == '__main__':
    validation_boards, validation_moves = my_get_data('./data/valid.xml')
    for epoch in range(num_of_epochs):
        print("epoch number: " + str(epoch + 1))
        for t in transformations:
            # boards, moves = my_get_data('./data/train.xml', t)
            # permuted_indices = np.random.permutation(boards.shape[0])
            # for i in tqdm(range(0, boards.shape[0], batch_size)):
            #     selected_data_points = np.take(permuted_indices, range(i, i+batch_size), mode='wrap')
            #     sess.run(train_op, {x: boards[selected_data_points], y: moves[selected_data_points], is_training: True})
            # training_accuracy = compute_accuracy(get_predictions(boards), np.argmax(moves, 1))
            # print("training accuracy: " + str(round(training_accuracy, 2)))
            # TODO ask about all transformations of validation boards and take the most probable answer
            validation_accuracy = compute_accuracy(get_predictions(validation_boards), np.argmax(validation_moves, 1))
            print("validation accuracy: " + str(round(validation_accuracy, 2)))
