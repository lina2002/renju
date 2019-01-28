import functools
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET


def _move_to_tuple(move):
    return ord(move[0]) - ord('a'), int(move[1:]) - 1


def _get_move_board(move, fill=1.0):
    move = _move_to_tuple(move)
    res = np.zeros((15, 15))
    res[move[0], move[1]] = fill
    return res


def _get_games(path):
    xml = ET.parse(path).getroot()
    for game in xml.iter('game'):
        yield game.text
        break


def get_data(path, color=None):
    """
    :param path: path to xml file
    :param color: if None will return all moves,
                  if 'black' returns only moves of black player,
                  if 'white' returns only moves of black player
    :return: tuple of two arrays of numpy arrays:
                  first with boards (matrix 15x15 with -1, 0 and 1 for white stone, empty and black stone, respectively)
                  second with moves (vector with size 15x15 and 1 on place where player put his stone).
    """
    boards = []
    moves = []

    start_pos = 6
    step = 1
    if color is not None:
        start_pos += 0 if color.lower() == 'black' else 1
        step = 2

    for game in _get_games(path):
        game = game.split()
        board = np.zeros((15, 15))
        for i, (move, next_move) in enumerate(zip(game[:-1], game[1:])):
            board = board + _get_move_board(move, (-1)**i)
            move = _get_move_board(next_move).flatten()
            if i >= start_pos and (i - start_pos)%step == 0:
                boards.append(board)
                moves.append(move)

    return boards, moves

#  czarne: 0, 2, 4, 6
#  białe: 1, 3, 5, 7
#  zaczynamy zgadywania ruchu od białego, 8. ruchu w grze


def my_get_data(path):
    """
    :param path: path to xml file
    :param color: if None will return all moves,
                  if 'black' returns only moves of black player,
                  if 'white' returns only moves of black player
    :return: tuple of two arrays of numpy arrays:
                  first with boards (matrix 15x15 with -1, 0 and 1 for white stone, empty and black stone, respectively)
                  second with moves (vector with size 15x15 and 1 on place where player put his stone).
    """
    boards = []
    moves = []

    start_pos = 6

    for game in _get_games(path):
        game = game.split()
        board = np.zeros((15, 15))
        for i, (move, next_move) in enumerate(zip(game[:-1], game[1:])):
            board = board + _get_move_board(move, -2*(i % 2) + 1)
            move = _get_move_board(next_move).flatten()
            if i >= start_pos:
                if i % 2 == 0:
                    black_is_current = zeros()
                    my_stones = white_fields(board)
                    opponent_stones = black_fields(board)
                else:
                    black_is_current = ones()
                    my_stones = black_fields(board)
                    opponent_stones = white_fields(board)
                stacked = np.stack((my_stones, opponent_stones, empty_fields(board), black_is_current, ones()), axis=2)
                boards.append(stacked)
                moves.append(move)

    return np.asarray(boards), np.asarray(moves)


def empty_fields(board):
    is_empty = lambda field: field == 0
    return is_empty(board).astype(float)


def black_fields(board):
    is_black = lambda field: field == 1
    return is_black(board).astype(float)


def white_fields(board):
    is_white = lambda field: field == -1
    return is_white(board).astype(float)


def zeros():
    return np.zeros((15, 15))


def ones():
    return np.ones((15, 15))


np.set_printoptions(threshold=np.nan)


def compute_accuracy(predictions, labels):
    correctly_predicted = np.sum(predictions == labels)
    all = labels.shape[0]
    return 100*correctly_predicted/all


batch_size = 32
num_of_epochs = 100
is_training = tf.placeholder_with_default(False, shape=(), name='is_training')


N = 128


x = tf.placeholder(tf.float32, [None, 15, 15, 5])
y = tf.placeholder(tf.float32, [None, 225])


my_batchn = functools.partial(
    tf.layers.batch_normalization,
    training=is_training)

my_conv2d = functools.partial(
    tf.layers.conv2d,
    filters=N,
    padding="same",
    data_format="channels_last",
    use_bias=False)


def my_res_layer(inputs):
    int_layer1 = tf.nn.relu(my_conv2d(inputs, kernel_size=7))
    initial_output = my_batchn(int_layer1)
    int_layer2 = my_conv2d(initial_output, kernel_size=7)
    output = tf.nn.relu(inputs + int_layer2)
    return output


z = tf.pad(x, tf.constant([[0, 0], [0, 0], [0, 0], [0, N-5]]))

for _ in range(6):
    z = my_res_layer(z)

z = my_batchn(z)
z = tf.reshape(z, (-1, 15*15*N))
z = tf.layers.dense(z, 3072)
z = tf.nn.relu(z)
z = tf.layers.dense(z, 225)
z = my_batchn(z)


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


my_rot = functools.partial(np.rot90, axes=(1, 2))
my_rot_180 = compose2(my_rot, my_rot)
my_rot_270 = compose3(my_rot, my_rot, my_rot)

my_flip = functools.partial(np.flip, axis=2)

my_rot_flip = compose2(my_rot, my_flip)
my_rot_180_flip = compose2(my_rot_180, my_flip)
my_rot_270_flip = compose2(my_rot_270, my_flip)


transformations = [identity, my_rot, my_rot_180, my_rot_270, my_flip, my_rot_flip, my_rot_180_flip, my_rot_270_flip]
inverse_transformations = [identity, my_rot_270, my_rot_180, my_rot, my_flip, my_rot_flip, my_rot_180_flip, my_rot_270_flip]


def transform_vector(t, v):
    v = np.reshape(v, (-1, 15, 15))
    v = t(v)
    v = np.reshape(v, (-1, 225))
    return v


def predict(bs, c):
    bsr = reverse_colors(bs)
    inputs = [t(bs) for t in transformations]
    inputs_reversed = [t(bsr) for t in transformations]
    predictions = [y_pred.eval({x: inp}) for inp in inputs]
    predictions_reversed = [y_pred.eval({x: inp}) for inp in inputs_reversed]

    for i, t in enumerate(inverse_transformations):
        predictions[i] = transform_vector(t, predictions[i])
        predictions_reversed[i] = transform_vector(t, predictions_reversed[i])

    predictions = sum(predictions)
    predictions_reversed = sum(predictions_reversed)
    return np.argmax(predictions + c*predictions_reversed, 1)


def reverse_colors(bs):
    bsr = bs.copy()
    bsr[:,:,:,1] = bs[:,:,:,0]
    bsr[:,:,:,0] = bs[:,:,:,1]
    bsr[:,:,:,3] = 1 - bsr[:,:,:,3]
    return bsr


def get_predictions(bs, c):
    preditions = []
    for i in tqdm(range(0, bs.shape[0], 100)):
        preditions = np.append(preditions, predict(bs[i:(i+100)], c))
    return preditions


def predict_training(bs):
    return np.argmax(y_pred.eval({x: bs}), 1)


def get_predictions_training(bs):
    preditions = []
    for i in tqdm(range(0, bs.shape[0], 100)):
        preditions = np.append(preditions, predict_training(bs[i:(i+100)]))
    return preditions


def transform_vector_2d(t, v):
    v = np.reshape(v, (15, 15))
    v = t(v)
    v = np.reshape(v, 225)
    return v


if __name__ == '__main__':
    boards, moves = my_get_data('./data/train.xml')
    validation_boards, validation_moves = my_get_data('./data/valid.xml')
    rot_180 = compose2(np.rot90, np.rot90)
    rot_270 = compose3(np.rot90, np.rot90, np.rot90)
    rot_flip = compose2(np.rot90, np.fliplr)
    rot_180_flip = compose2(rot_180, np.fliplr)
    rot_270_flip = compose2(rot_270, np.fliplr)
    transformations_2d = [identity, np.rot90, rot_180, rot_270, np.fliplr, rot_flip, rot_180_flip, rot_270_flip]
    for epoch in range(num_of_epochs):
        print("epoch number: " + str(epoch + 1))
        permuted_indices = np.random.permutation(boards.shape[0])
        for i in tqdm(range(0, boards.shape[0], batch_size)):
            selected_data_points = np.take(permuted_indices, range(i, i+batch_size), mode='wrap')
            for t in transformations:
                sess.run(train_op, {x: t(boards[selected_data_points]), y: transform_vector(t, moves[selected_data_points]), is_training: True})
        training_accuracy = compute_accuracy(get_predictions_training(boards), np.argmax(moves, 1))
        print("training accuracy: " + str(round(training_accuracy, 2)))
        validation_accuracy = compute_accuracy(get_predictions(validation_boards, 0.1), np.argmax(validation_moves, 1))
        print("validation accuracy: " + str(round(validation_accuracy, 2)))
