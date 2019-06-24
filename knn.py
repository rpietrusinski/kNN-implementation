

class KnnClassifier(object):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    def standardize(self, vector):
        mean = np.mean(vector)
        sd = np.std(vector)
        return np.array([(x - mean) / sd for x in vector])

    def prepare_data(self):
        X_scaled = np.apply_along_axis(self.standardize, 0, self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=0.33,
                                                                                random_state=42)

    def majority_vote(self, lst):
        return max(set(lst), key=lst.count)

    def single_pred(self, vec, neighbors):
        most_similar_indices = [i[0] for i in sorted(enumerate(vec), key=lambda x: x[1])[:neighbors]]
        prediction = self.majority_vote(list(self.y_train[most_similar_indices]))
        return prediction

    def predict(self, test_data, neighbors):
        obs_train = self.X_train.shape[0]
        obs_test = test_data.shape[0]

        # distance matrix
        distance_matrix = np.zeros((obs_train, obs_test))
        distance_matrix[:] = np.nan
        for i in range(obs_train):
            for j in range(obs_test):
                vec_i = self.X_train[i, :]
                vec_j = test_data[j, :]
                euc = np.sqrt(np.sum([(p - q) ** 2 for p, q in zip(vec_i, vec_j)]))
                distance_matrix[i, j] = euc
        return np.apply_along_axis(self.single_pred, 0, distance_matrix, neighbors)
