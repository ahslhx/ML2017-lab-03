import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        print('Initialize AdaBoostClassifier')
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier_list = []
        self.alpha_list = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples, n_features = X.shape
        distribution = np.ones(n_samples) * (1 / n_samples)
        print(distribution.shape)

        for i in range(self.n_weakers_limit):
            print('Building No.', i + 1, 'Weak Classifier...')
            weak_classifier = self.weak_classifier
            weak_classifier.fit(X, y, sample_weight=distribution)
            self.weak_classifier_list.append(weak_classifier)
            pred_y = weak_classifier.predict(X)
            error = np.sum(distribution * (pred_y != y))
            alpha = np.log((1.0-error)/max(error,1e-16))/2.0
            self.alpha_list.append(alpha)

            for j in range(n_samples):
                distribution[j] = distribution[j] * np.exp(-alpha * y[j] * pred_y[j])
            distribution = distribution / np.sum(distribution)

        return self


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of different samples, which shape should be (n_samples,1).
        '''
        n_samples = X.shape[0]
        n_weakers_limit = self.n_weakers_limit
        score = np.zeros((n_samples, 1))
        for i in range(n_weakers_limit):
            score_i = self.weak_classifier_list[i].predict(X).reshape(n_samples, 1)
            score = score + self.alpha_list[i] * score_i

        return score


    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        n_samples = X.shape[0]
        score = self.predict_scores(X)
        pred_y = np.ones((n_samples, 1))
        pred_y[score < threshold] = -1

        return pred_y


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
