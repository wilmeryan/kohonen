import numpy as np
from som_network import SomNetwork
from som_network_jit import SomNetworkJit


class TestSom:
    @classmethod
    def setup_class(cls):
        cls.som_5x5 = SomNetwork(map_size=(5, 5), input_dim=3)
        cls.som_20x20 = SomNetwork(map_size=(20, 20), input_dim=3)
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
        self.som_20x20.weights[3, 3, :] = np.array([0, 1, 2])
        np.testing.assert_almost_equal(self.som_20x20.get_bmu_coord(self.input_vec_1), (3, 3))


class TestSomJit:
    @classmethod
    def setup_class(cls):
        cls.som_5x5 = SomNetworkJit(map_size=(5, 5), input_dim=3)
        cls.som_5x5_slow = SomNetwork(map_size=(5, 5), input_dim=3)

        cls.som_20x20 = SomNetworkJit(map_size=(20, 20), input_dim=3)
        cls.som_20x0_slow = SomNetwork(map_size=(5, 5), input_dim=3)

        cls.som_20x20 = SomNetworkJit(map_size=(20, 20), input_dim=3)
        cls.som_20x0_slow = SomNetwork(map_size=(5, 5), input_dim=3)

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
        self.som_20x20.weights[3, 3, :] = np.array([0, 1, 2])
        np.testing.assert_almost_equal(self.som_20x20.get_bmu_coord(self.input_vec_1), (3, 3))

    # We can compare that the fitting logic is the same as the original implementation
    def compare_with_slow_small(self):
        # set the weights the same
        self.som_5x5.weights = self.som_5x5_slow.weights
        slow = self.som_5x5.fit(input_matrix=self.input_vec_1, n_iterations=5)
        jit = self.som_5x5_slow.fit(input_matrix=self.input_vec_1, n_iterations=5)
        np.test.assert_almost_equal(slow, jit)

    def compare_with_slow_big(self):
        # set the weights the same
        self.som_20x20.weights = self.som_20x20_slow.weights
        slow = self.som_20x20.fit(input_matrix=self.input_vec_1, n_iterations=100)
        jit = self.som_20x20_slow.fit(input_matrix=self.input_vec_1, n_iterations=100)
        np.test.assert_almost_equal(slow, jit)