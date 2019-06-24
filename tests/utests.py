import unittest
from knn import KnnClassifier
import numpy as np


class TestKnn(unittest.TestCase):

    def test_standardize(self):
        X = np.array(np.random.sample(50)).reshape(10, 5)
        y = np.random.randint(0, 2, 50).reshape(50, 1)
        knn = KnnClassifier(X, y)
        self.assertIsInstance(knn.standardize(X[:, 0]), np.ndarray)
        self.assertIsNotNone(knn.standardize(X[:, 0]))

    def test_majority_vote(self, lst):
        X = np.array(np.random.sample(50)).reshape(10, 5)
        y = np.random.randint(0, 2, 50).reshape(50, 1)
        knn = KnnClassifier(X, y)
        self.assertIsInstance(knn.majority_vote([1, 2, 3]), int)
        self.assertIsNotNone(knn.majority_vote([1, 2, 3, 3]))

if __name__=='__main__':
    unittest.main()


