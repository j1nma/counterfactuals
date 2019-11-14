import numpy as np


def ite(y_treated, y_controlled):
    """ Individualized Treatment Effect """
    """ Measures performance of the treatment on a particular study individual. """
    """ - y_treated: predicted outcome with T = 1
        - y_controlled: predicted outcome with T = 0
    """
    return y_treated - y_controlled


def ate(y_treated_predicted, y_controlled_predicted):
    """ Average Treatment Effect """
    """ Average of ITE on a population of individuals. """
    """ - y_treated_predicted: predicted outcome with T = 1 for a population
        - y_controlled_predicted: predicted outcome with T = 0 or a population
    """

    return np.mean(y_treated_predicted - y_controlled_predicted)


def pehe(y_treated_predicted, y_controlled_predicted, y_treated_true, y_controlled_true):
    """ Precision in Estimation of Heterogeneous Effect """
    """ Measures how accurate the predictions of both the factual and counterfactual regimes are. """
    """ - y_treated_predicted: predicted outcome with T = 1 for a population
        - y_controlled_predicted: predicted outcome with T = 0 or a population
        - y_treated_true: true outcome with T = 1 for a population
        - y_controlled_true: true outcome with T = 0 or a population
    """

    return np.sqrt(
        np.mean(np.square((y_treated_true - y_controlled_true) - (y_treated_predicted - y_controlled_predicted))))


def abs_ate(y_treated_predicted, y_controlled_predicted, y_treated_true, y_controlled_true):
    """ Absolute Error of Average Treatment Effect """
    """ Error between the average of ITE on a population of individuals and the true ITE. """
    """ - y_treated_predicted: predicted outcome with T = 1 for a population
        - y_controlled_predicted: predicted outcome with T = 0 or a population
        - y_treated_true: true outcome with T = 1 for a population
        - y_controlled_true: true outcome with T = 0 or a population
    """
    return np.abs(np.mean(y_treated_predicted - y_controlled_predicted) - np.mean(y_treated_true - y_controlled_true))


def rmse_ite(t, y_factual, y_treated_predicted, y_controlled_predicted, y_treated_true, y_controlled_true):
    """ Root of the Mean Square Error of the Individualized Treatment Effect """
    """ - t: observed treatments
        - y_factual: observed outcomes
        - y_treated_predicted: predicted outcome with T = 1 for a population
        - y_controlled_predicted: predicted outcome with T = 0 or a population
        - y_treated_true: true outcome with T = 1 for a population
        - y_controlled_true: true outcome with T = 0 or a population
    """

    true_ITE = y_treated_true - y_controlled_true
    predicted_ITE = np.zeros_like(true_ITE)

    treated_indexes = np.where(t == 1)[0]
    controlled_indexes = np.where(t == 0)[0]

    predicted_ITE[treated_indexes] = ite(y_factual[treated_indexes], y_controlled_predicted[treated_indexes])
    predicted_ITE[controlled_indexes] = ite(y_treated_predicted[controlled_indexes], y_factual[controlled_indexes])

    return np.sqrt(np.mean(np.square(predicted_ITE - true_ITE)))


def rmse_factual(y_factual, y_factual_predicted):
    """ Root of the Mean Square Error between the factual prediction and the observed outcome """
    """ - y_factual: observed outcome
        - y_factual_predicted: predicted outcome of the observed outcome
    """
    return np.sqrt(np.mean(np.square(y_factual_predicted - y_factual)))


def rmse_counterfactual(y_counterfactual, y_counterfactual_predicted):
    """ Root of the Mean Square Error between the counterfactual prediction and the observed outcome """
    """ - y_counterfactual: counterfactual outcome
        - y_counterfactual_predicted: predicted outcome of the counterfactual outcome
    """
    return np.sqrt(np.mean(np.square(y_counterfactual_predicted - y_counterfactual)))


def get_errors(t, y_observed, y_treated_predicted, y_controlled_predicted, y_treated_true, y_controlled_true):
    rmse_ite_ = rmse_ite(t, y_observed, y_treated_predicted, y_controlled_predicted, y_treated_true,
                         y_controlled_true)

    abs_ate_ = abs_ate(y_treated_predicted, y_controlled_predicted, y_treated_true, y_controlled_true)

    pehe_ = pehe(y_treated_predicted, y_controlled_predicted, y_treated_true, y_controlled_true)

    return rmse_ite_, abs_ate_, pehe_
