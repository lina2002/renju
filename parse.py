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
#  to może trochę zmieniać

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

    return boards, moves


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


if __name__ == '__main__':
    boards, moves = my_get_data('./data/valid.xml')
    print(moves[0].shape)
