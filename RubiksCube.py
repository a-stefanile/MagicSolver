import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product
import random



faces = ['top', 'left', 'front', 'bottom', 'right', 'back']
possible_moves = tuple(product(faces, [False, True]))

class RubiksCube:
    def __init__(self):
        # Initialize a 3D tensor to represent the Rubik's Cube
        self.cube = np.empty((5, 5, 5), dtype='U10')
        self.cube[:, :, :] = ''

        # Initialize sticker colors
        self.cube[0, 1:-1, 1:-1] = 'w'  # Top (white)
        self.cube[1:-1, 0, 1:-1] = 'g'  # Front (green)
        self.cube[1:-1, 1:-1, 0] = 'r'  # Left (red)
        self.cube[-1, 1:-1, 1:-1] = 'y' # Bottom (yellow)
        self.cube[1:-1, -1, 1:-1] = 'b' # Back (blue)
        self.cube[1:-1, 1:-1, -1] = 'o' # Right (orange)

    def print_cube(self):
        print(self.cube)

    def rotate_face(self, face, reverse=False):
        """
        Rotates a given face of the cube 90 degrees.

        Parameters:
            face (str): One of ['top', 'front', 'left', 'bottom', 'back', 'right']
            reverse (bool): if the rotation should be reversed
        """
        # maps a face to the section of the tensor which needs to be rotated
        rot_map = {
            'top': (slice(0, 2), slice(0, 5), slice(0, 5)),
            'left': (slice(0, 5), slice(0, 2), slice(0, 5)),
            'front': (slice(0, 5), slice(0, 5), slice(0, 2)),
            'bottom': (slice(3, 5), slice(0, 5), slice(0, 5)),
            'right': (slice(0, 5), slice(3, 5), slice(0, 5)),
            'back': (slice(0, 5), slice(0, 5), slice(3, 5))
        }

        # getting all of the stickers that will be rotating
        rotating_slice = self.cube[rot_map[face]]

        # getting the axis of rotation
        axis_of_rotation = np.argmin(rotating_slice.shape)

        # rotating about axis of rotation
        axes_of_non_rotation = [0,1,2]
        axes_of_non_rotation.remove(axis_of_rotation)
        axes_of_non_rotation = tuple(axes_of_non_rotation)
        direction = 1 if reverse else -1
        rotated_slice = np.rot90(rotating_slice, k=direction, axes=axes_of_non_rotation)

        # overwriting cube
        self.cube[rot_map[face]] = rotated_slice

    def _rotate_cube_180(self):
        """
        Rotate the entire cube 180 degrees by flipping and transposing
        this is used for visualization
        """
        # Rotate the cube 180 degrees
        rotated_cube = np.rot90(self.cube, k=2, axes=(0,1))
        rotated_cube = np.rot90(rotated_cube, k=1, axes=(1,2))
        return rotated_cube

    def visualize_opposite_corners(self, return_fig = False):
        """
        Visualize the Rubik's Cube from two truly opposite corners
        """
        # Create a new figure with two subplots
        fig = plt.figure(figsize=(20, 10))

        # Color mapping
        color_map = {
            'w': 'white',
            'g': 'green',
            'r': 'red',
            'y': 'yellow',
            'b': 'blue',
            'o': 'orange'
        }

        # Cubes to visualize: original and 180-degree rotated
        cubes_to_render = [
            {
                'cube_data': self.cube,
                'title': 'View 1'
            },
            {
                'cube_data': self._rotate_cube_180(),
                'title': 'View 2'
            }
        ]

        # Create subplots for each view
        for i, cube_info in enumerate(cubes_to_render, 1):
            ax = fig.add_subplot(1, 2, i, projection='3d')

            ax.view_init(elev=-150, azim=45, vertical_axis='x')

            # Iterate through the cube and plot non-empty stickers
            cube_data = cube_info['cube_data']
            for x in range(cube_data.shape[0]):
                for y in range(cube_data.shape[1]):
                    for z in range(cube_data.shape[2]):
                        # Only plot if there's a color
                        if cube_data[x, y, z] != '':
                            color = color_map.get(cube_data[x, y, z], 'gray')

                            # Define the 8 vertices of the small cube
                            vertices = [
                                [x, y, z], [x+1, y, z],
                                [x+1, y+1, z], [x, y+1, z],
                                [x, y, z+1], [x+1, y, z+1],
                                [x+1, y+1, z+1], [x, y+1, z+1]
                            ]

                            # Define the faces of the cube
                            faces = [
                                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                                [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
                            ]

                            # Plot each face
                            for face in faces:
                                poly = Poly3DCollection([face], alpha=1, edgecolor='black')
                                poly.set_color(color)
                                poly.set_edgecolor('black')
                                ax.add_collection3d(poly)

            # Set axis limits and equal aspect ratio
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.set_zlim(0, 5)
            ax.set_box_aspect((1, 1, 1))

            # Remove axis labels and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_title(cube_info['title'])

        plt.tight_layout()

        #return figure instead of rendering it
        if return_fig:
            return fig

        plt.show()

    def get_state(self):
        # 1. Estrazione dei 54 sticker (come fatto prima)
        faces = [
            self.cube[0, 1:4, 1:4],  # Top
            self.cube[-1, 1:4, 1:4],  # Bottom
            self.cube[1:4, 0, 1:4],  # Front
            self.cube[1:4, -1, 1:4],  # Back
            self.cube[1:4, 1:4, 0],  # Left
            self.cube[1:4, 1:4, -1]  # Right
        ]
        flat_stickers = np.array(faces).flatten()

        # 2. Definizione dell'ordine dei colori
        colors = ['w', 'y', 'g', 'b', 'r', 'o']

        # 3. Creazione della matrice One-Hot
        # Per ogni sticker, creiamo un array di 6 booleani/interi
        ohe_matrix = np.zeros((len(flat_stickers), len(colors)), dtype=int)

        for i, sticker in enumerate(flat_stickers):
            if sticker in colors:
                color_index = colors.index(sticker)
                ohe_matrix[i, color_index] = 1

        # 4. Flattening finale per avere un unico vettore 1D di 324 elementi
        return ohe_matrix.flatten()

    def scramble(self, n=20):
        moves = []
        for _ in range(n):
            # selecting a random move
            selected_move = random.choice(possible_moves)
            moves.append(selected_move)

            # Rotate a face to show some variation
            self.rotate_face(selected_move[0], reverse=selected_move[1])
        return moves

    def unscramble(self, moves):
        moves.reverse()
        for i in range(20):
            selected_move = moves[i]
            self.rotate_face(selected_move[0], reverse=not selected_move[1])

    def get_manhattan_features(self):
        faces_data = [
            self.cube[0, 1:4, 1:4],  # Top
            self.cube[-1, 1:4, 1:4],  # Bottom
            self.cube[1:4, 0, 1:4],  # Front
            self.cube[1:4, -1, 1:4],  # Back
            self.cube[1:4, 1:4, 0],  # Left
            self.cube[1:4, 1:4, -1]  # Right
        ]
        flat_stickers = np.array(faces_data).flatten()
        color_to_target_face = {'w': 0, 'y': 1, 'g': 2, 'b': 3, 'r': 4, 'o': 5}
        current_face_indices = np.repeat(np.arange(6), 9)

        target_face_indices = np.array([color_to_target_face[c] for c in flat_stickers])
        opposites = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

        distances = np.zeros(54, dtype=np.uint8)
        diff_mask = target_face_indices != current_face_indices
        distances[diff_mask] = 1
        for f1, f2 in opposites.items():
            opposite_mask = (current_face_indices == f1) & (target_face_indices == f2)
            distances[opposite_mask] = 2
        return distances

    def is_solved(self):
        """Verifica se il cubo è in stato risolto."""
        # Se tutti i quadratini sono alla distanza 0 dalla loro faccia, è risolto
        return np.all(self.get_manhattan_features() == 0)

if __name__ == "__main__":
    cube=RubiksCube()
    cube.visualize_opposite_corners()