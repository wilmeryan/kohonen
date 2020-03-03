import numpy as np
from som_network import SomNetwork
from som_network_jit import SomNetworkJit


# class TestSomJit:
#     @classmethod
#     def setup_class(cls):
#         cls.som_5x5 = SomNetworkJit(map_size=(5, 5), input_dim=3)
#         cls.som_5x5_slow = SomNetwork(map_size=(5, 5), input_dim=3)
#
#         cls.som_20x20 = SomNetworkJit(map_size=(20, 20), input_dim=3)
#         cls.som_20x0_slow = SomNetwork(map_size=(5, 5), input_dim=3)
#
#         cls.som_20x20 = SomNetworkJit(map_size=(20, 20), input_dim=3)
#         cls.som_20x20_slow = SomNetwork(map_size=(5, 5), input_dim=3)
#
#         cls.som_100x100 = SomNetworkJit(map_size=(20, 20), input_dim=3)
#         cls.som_100x100_small = SomNetwork(map_size=(20, 20), input_dim=3)
#
#         cls.input_vec_1 = np.arange(0, 3).reshape((1, -1))
#         cls.input_vec_2 = np.arange(0, 6).reshape((2, -1))
#
#     def test_weights_size(self):
#         assert self.som_5x5.weights.shape == (5, 5, 3)
#
#     def test_neighbourhood_mask(self):
#         mask = self.som_5x5.get_neighbourhood_mask(point_coord=(1, 1), radius=2)
#         mask_gt = np.array([[True, True, True, False, False],
#                             [True, True, True, True, False],
#                             [True, True, True, False, False],
#                             [False, True, False, False, False],
#                             [False, False, False, False, False]])
#         np.testing.assert_array_equal(mask, mask_gt)
#
#     def test_bmu_coord(self):
#         self.som_20x20.weights[3, 3, :] = np.array([0, 1, 2])
#         np.testing.assert_almost_equal(self.som_20x20.get_bmu_coord(self.input_vec_1), (3, 3))
#
#     # We can compare that the fitting logic is the same as the original implementation
#     def test_compare_with_slow_small(self):
#         # set the weights the same
#         self.som_5x5.weights = self.som_5x5_slow.weights
#         slow = self.som_5x5.fit(input_matrix=self.input_vec_1, n_iterations=5)
#         jit = self.som_5x5_slow.fit(input_matrix=self.input_vec_1, n_iterations=5)
#         np.test.assert_almost_equal(slow, jit)
#
#     def test_compare_with_slow_big(self):
#         # set the weights the same
#         self.som_20x20.weights = self.som_20x20_slow.weights
#         slow = self.som_20x20.fit(input_matrix=self.input_vec_1, n_iterations=100)
#         jit = self.som_20x20_slow.fit(input_matrix=self.input_vec_1, n_iterations=100)
#         np.testing.assert_almost_equal(slow, jit)
#
#         self.som_100x100.weights = self.som_100x100_small.weights
#         slow = self.som_100x100.fit(input_matrix=self.input_vec_1, n_iterations=100)
#         jit = self.som_100x100_slow.fit(input_matrix=self.input_vec_1, n_iterations=100)
#         np.testing.assert_almost_equal(slow, jit)

class TestSomJit:
    @classmethod
    def setup_class(cls):
        cls.som_5x5 = SomNetworkJit(map_size=(5, 5), input_dim=3)
        cls.som_20x20 = SomNetworkJit(map_size=(20, 20), input_dim=3)
        cls.input_vec_1 = np.arange(0, 3).reshape((1, -1))
        cls.input_vec_2 = np.arange(0, 6).reshape((2, -1))

    def test_weights_size(self):
        assert self.som_5x5.weights.shape == (5, 5, 3)

    def test_neighbourhood_mask(self):
        mask = self.som_5x5.get_neighbourhood_mask(point_coord=(1, 1), radius=2)
        mask_gt = np.array([[True, True, True, False, False],
                            [True, True, True, True, False],
                            [True, True, True, False, False],
                            [False, True, False, False, False],
                            [False, False, False, False, False]])
        np.testing.assert_array_equal(mask, mask_gt)

    def test_bmu_coord(self):
        self.som_20x20.weights[10, 3, :] = np.array([0, 1, 2])
        np.testing.assert_almost_equal(self.som_20x20.get_bmu_coord(self.input_vec_1), (10, 3))

        self.som_20x20.weights[16, 2, :] = np.array([3, 4, 5])
        np.testing.assert_almost_equal(self.som_20x20.get_bmu_coord(self.input_vec_2[1]), (16, 2))

    def test_neighbourhood_radius(self):
        np.testing.assert_array_equal(self.som_20x20.calc_neighbourhood_radius(5, 20),
                                      10 * np.exp(-5 / (20 / np.log(10))))

    def test_learning_rate(self):
        np.testing.assert_array_equal(self.som_20x20.get_learning_rate(5, 20), 0.1 * np.exp(-(5 / (20 / np.log(10)))))

    def test_bmu_matrix(self):
        self.som_5x5.weights = np.zeros((5, 5, 3))

        # Set 0,0 -> [0,1,2] and 3,3 -> 3, 4, 5
        # 3, 3 ->
        self.som_5x5.weights[0, 0, :] = np.array([0, 1, 2])
        self.som_5x5.weights[3, 3, :] = np.array([3, 4, 5])
        answer_array = np.ones((5, 5)) * np.sqrt(np.sum(np.array([0, 1, 4])))
        answer_array[0, 0] = np.sqrt(np.sum(np.array([0, 0, 0])))
        answer_array[3, 3] = np.sqrt(np.sum(np.array([9, 9, 9])))
        np.testing.assert_almost_equal(self.som_5x5.calc_bmu_matrix_vectorised(self.som_5x5.weights, self.input_vec_1), answer_array)

    def test_influence_matrix(self):
        neighbourhood_radius = 3
        point_coord = (1, 1)
        influence_matrix = self.som_5x5.get_influence_matrix(self.som_5x5.weights, point_coord, neighbourhood_radius)
        dist_matrix = np.array([[np.sqrt(2), 1, np.sqrt(2), np.sqrt(5), np.sqrt(10)],
                                [1, 0, 1, 2, 3],
                                [np.sqrt(2), 1, np.sqrt(2), np.sqrt(5), np.sqrt(10)],
                                [np.sqrt(5), 2, np.sqrt(5), np.sqrt(8), np.sqrt(13)],
                                [np.sqrt(10), 3, np.sqrt(10), np.sqrt(13), np.sqrt(18)]])
        ans = np.array(np.exp(-np.square(dist_matrix) / (2 * np.square(neighbourhood_radius))))
        np.testing.assert_almost_equal(ans, influence_matrix)

