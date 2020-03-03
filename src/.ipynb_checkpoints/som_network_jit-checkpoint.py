from tqdm import tqdm
import numpy as np
from numba import jit


class SomNetworkJit:
    def __init__(self, map_size=(10, 10), input_dim=3):
        """Kohonen's Self Organising Map Class.

        This class implements the SOM in a mostly vectorized way in numpy.

        :param map_size (int, int): The size of the weights map of the SOM network
        :param input_dim (int): The dimension of the input vector
        """
        self.weights = np.random.random(size=(map_size[0], map_size[1], input_dim)).astype(np.float64)
        self.init_rad = max(map_size[0], map_size[1]) / 2
        self.init_learning_rate = 0.1
        self.fitted = False

    @staticmethod
    @jit(nopython=True, parallel=False)
    def calc_bmu_matrix_vectorised(matrix, vec_a):
        """Calculates the BMU euclidean distance from the input vector for each node on the SOM map.

        :param vec_a (np.array): The input vector vec_a
        :return: np.array matrix for the euclidean distance from the input vector for each node on the SOM map
        """
        subtracted_mat = np.multiply(np.ones((matrix.shape[0], matrix.shape[1], matrix.shape[2])), vec_a) - matrix
        squared = np.multiply(subtracted_mat, subtracted_mat)
        return np.sqrt(np.sum(squared, axis=-1))

    def get_bmu_coord(self, vec_a):
        """Gets the BMU coord from the self.weights matrix based on the input vector vec_a

        :param vec_a (np.array):
        :return: (int, int) coordinates of the BMU of the SOM weights
        """
        bmu_matrix = self.calc_bmu_matrix_vectorised(self.weights, vec_a)
        # Returns the flattened argmin index
        bmu_index = np.argmin(bmu_matrix, axis=None)
        # Need to get the 0th and 1st indexes from the flattened
        bmu_index_0 = bmu_index // self.weights.shape[0]
        bmu_index_1 = bmu_index % self.weights.shape[0]
        return (bmu_index_0, bmu_index_1)

    def get_time_const(self, n_iterations):
        """Calculates the time const based on the iteration number"""
        return n_iterations / np.log(self.init_rad)

    def calc_neighbourhood_radius(self, iteration_no, n_iterations):
        """ Calculates the the neighbourhood radius

        :param iteration_no (int): The iteration number
        :param n_iterations (int): the number of iterations
        :return: float of neighbourhood radius
        """
        return self.init_rad * np.exp(-(iteration_no / self.get_time_const(n_iterations)))

    def get_neighbourhood_mask(self, point_coord, radius):
        """ Gets the neighbour radius mask for a speicifed point_coord

        :param point_coord tuple(int,int): the center coordinates to get the neighbour radius from
        :param radius: the radius to get the mask from
        :return: np.array binary mask of points which are True for radius dist away from point_coord
        """
        y, x = np.ogrid[-point_coord[0]:self.weights.shape[0] - point_coord[0],
               -point_coord[1]:self.weights.shape[1] - point_coord[1]]
        mask = x * x + y * y <= radius * radius
        return mask

    def get_learning_rate(self, iteration_no, n_iterations):
        """ Returns the learning rate from the iteration_no and n_iterations

        :param iteration_no(int): The iteration number
        :param n_iterations(int): The number of iterations
        :return: float of the learning rate
        """
        return self.init_learning_rate * np.exp(-(iteration_no / self.get_time_const(n_iterations)))

    @staticmethod
    @jit(nopython=True)
    def get_influence_matrix(matrix, point_coord, neighbourhood_radius):
        """Calculates and returns the influence matrix from a specified point_coord and neighbourhood radius

        :param point_coord(tuple(int, int)): the point coordinates to calculate the influence matrix from
        :param neighbourhood_radius (int): the neighbourhood radius
        :return: np.array of the influence numbers for each point in the matrix
        """
        influence_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
        # Create index matrix
        indices = np.zeros((2, matrix.shape[0], matrix.shape[1]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                indices[1, i, j] = j
                indices[0, i, j] = i
        indices = indices.astype(np.int32)
        influence_matrix = np.square(point_coord[0] - indices[0]) + np.square(point_coord[1] - indices[1])
        return np.exp(-(influence_matrix / (2 * np.square(neighbourhood_radius))))

    def update_weights(self, input_vec, iteration_no, n_iterations):
        """ Update the weights according to Kohonen's map. A binary mask is used mask out all elements outside the
        neighbourhood radius. This is multiplied by the influence matrix so that any elements outside the radius are
        unaffected by the weight updates.

        :param input_vec (np.array dim=1): the input vector
        :param iteration_no (int): the iteration number
        :param n_iterations (int): the total number of iterations
        :return: the weights matrix
        """
        bmu_coord = self.get_bmu_coord(input_vec)
        radius = self.calc_neighbourhood_radius(iteration_no, n_iterations)

        # We can use the mask times the influence matrix to set the new weights. Zero'd influence means no change
        mask = self.get_neighbourhood_mask(bmu_coord, radius).astype(int)
        influence_masked = self.get_influence_matrix(self.weights, bmu_coord, radius)
        subtracted_mat = np.multiply(np.ones((self.weights.shape[0], self.weights.shape[1], self.weights.shape[2])),
                                     input_vec) - self.weights
        lr_mult_influence_mask = np.multiply(self.get_learning_rate(iteration_no, n_iterations),
                                             influence_masked[:, :, None])
        second_term = np.multiply(lr_mult_influence_mask, subtracted_mat)
        self.weights = np.add(self.weights, second_term)

    def fit(self, input_matrix, n_iterations):
        """Fit the weights of the SOM network by iteratively looping through the matrix row by row for n_iterations.

        :param input_matrix (np.array): input matrix
        :param n_iterations (int): The number of iterations
        :return: np.array of the fitted SOM weights
        """

        assert not self.fitted, 'SOM has already been fitted'

        # iterate through each input vector
        for iteration_no in tqdm(range(n_iterations)):
            for input_vector in input_matrix:
                self.update_weights(input_vector, iteration_no, n_iterations)
        self.fitted = True
        return self.weights
