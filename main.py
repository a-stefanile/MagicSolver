import random
from itertools import product

from RubiksCube import RubiksCube
from array import *
faces = ['top', 'left', 'front', 'bottom', 'right', 'back']
possible_moves = tuple(product(faces, [False, True]))


def scramble(cube, n=20):
    moves = []
    for _ in range(n):
        # selecting a random move
        selected_move = random.choice(possible_moves)
        moves.append(selected_move)

        # Rotate a face to show some variation
        cube.rotate_face(selected_move[0], reverse=selected_move[1])
    return moves

def unscramble(cube, moves):
    moves.reverse()
    for i in range(20):
        selected_move = moves[i]
        cube.rotate_face(selected_move[0], reverse=not selected_move[1])


cube = RubiksCube()
moves = scramble(cube)
print(moves)
cube.visualize_opposite_corners()

unscramble(cube, moves)
cube.visualize_opposite_corners()
