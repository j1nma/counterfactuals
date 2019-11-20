import numpy as np


class Evaluator(object):
    def __init__(self, t, y_f, y_cf, mu0, mu1):
        self.t = t
        self.y_f = y_f
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        self.true_ite = mu1 - mu0

    def rmse_ite(self, y_treated_predicted, y_controlled_predicted):
        """ Root of the Mean Square Error of the Individualized Treatment Effect """
        """
            - ypred1: predicted outcome with T = 1 for a population
            - ypred0: predicted outcome with T = 0 or a population
        """
        predicted_ite = np.zeros_like(self.true_ite)
        treated_indices, controlled_indices = np.where(self.t == 1)[0], np.where(self.t == 0)[0]

        predicted_ite_for_treated = self.y_f[treated_indices] - y_controlled_predicted[treated_indices]
        predicted_ite_for_controlled = y_treated_predicted[controlled_indices] - self.y_f[controlled_indices]

        predicted_ite[treated_indices] = np.array([x.asscalar() for x in predicted_ite_for_treated])
        predicted_ite[controlled_indices] = np.array([x.asscalar() for x in predicted_ite_for_controlled])

        return np.sqrt(np.mean(np.square(self.true_ite - predicted_ite)))

    def abs_ate(self, y_treated_predicted, y_controlled_predicted):
        """ Absolute Error of Average Treatment Effect """
        """ Error between the average of ITE on a population of individuals and the true ITE. """
        """
            - y_treated_predicted: predicted outcome with T = 1 for a population
            - y_controlled_predicted: predicted outcome with T = 0 or a population
        """
        y_treated_predicted = np.array([x.asscalar() for x in y_treated_predicted])
        y_controlled_predicted = np.array([x.asscalar() for x in y_controlled_predicted])

        return np.abs(np.mean(y_treated_predicted - y_controlled_predicted) - np.mean(self.true_ite))

    def pehe(self, y_treated_predicted, y_controlled_predicted):
        """ Precision in Estimation of Heterogeneous Effect """
        """ Measures how accurate the predictions of both the factual and counterfactual regimes are. """
        """
            - y_treated_predicted: predicted outcome with T = 1 for a population
            - y_controlled_predicted: predicted outcome with T = 0 or a population
        """
        y_treated_predicted = np.array([x.asscalar() for x in y_treated_predicted])
        y_controlled_predicted = np.array([x.asscalar() for x in y_controlled_predicted])

        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (y_treated_predicted - y_controlled_predicted))))

    def get_metrics(self, y_treated_predicted, y_controlled_predicted):
        ite = self.rmse_ite(y_treated_predicted, y_controlled_predicted)
        ate = self.abs_ate(y_treated_predicted, y_controlled_predicted)
        pehe = self.pehe(y_treated_predicted, y_controlled_predicted)
        return ite, ate, pehe

    def get_rmse_f_cf(self, y_treated_predicted, y_controlled_predicted):
        """ Root of the Mean Square Error of the predicted factual and counterfactual outcomes """
        """
            - y_treated_predicted: predicted outcome with T = 1 for a population
            - y_controlled_predicted: predicted outcome with T = 0 or a population
        """
        y1 = y_treated_predicted
        y0 = y_controlled_predicted
        y_factual_predicted = self.t * y1 + (1 - self.t) * y0
        y_counterfactual_predicted = (1 - self.t) * y1 + self.t * y0

        rmse_factual = np.sqrt(np.mean(np.square(y_factual_predicted - self.y_f)))
        rmse_counterfactual = np.sqrt(np.mean(np.square(y_counterfactual_predicted - self.y_cf)))

        return rmse_factual, rmse_counterfactual
