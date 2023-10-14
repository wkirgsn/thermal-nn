import numpy as np
from sklearn.metrics import mean_squared_error as mse, mean_squared_log_error\
    as msle, mean_absolute_error as mae, r2_score
from tensorflow.keras import backend as K


def print_scores(y_true, y_pred):
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    metr_d = {'mse': (mse(y_true, y_pred), 'K²'),
              'mae': (mae(y_true, y_pred), 'K'),
              'l_infty': (np.max(np.abs(y_pred - y_true)), 'K'),
              'R2': (r2_score(y_true, y_pred), ''),
              'weighted_temp_mse': (temperature_loss_sklearn(y_true, y_pred),
                                    'K²')}
    ret = {}
    for k, (score, unit) in metr_d.items():
        print(f'{k}: {np.round(score, decimals=8):.6} {unit}')
        ret[k] = score
    return ret


def temperature_loss_keras(max_allowed_temperature=120,
                           min_occuring_temperature=0):
    def _wrapper(y_true, y_pred):
        k = np.sqrt(10)  # scale factor for under-estimates
        diff = y_true - y_pred

        # normalize on absolute temperature range
        l_bound = K.abs(min_occuring_temperature)
        diff *= (y_true + l_bound) / (max_allowed_temperature + l_bound)

        # lift weight for under-estimates
        diff *= K.maximum(K.cast(diff > 0, dtype=np.float32) * k, 1)

        return K.mean(K.square(diff), axis=-1)
    return _wrapper


def temperature_loss_sklearn(y_true, y_pred, sample_weight=None,
                             max_allowed_temperature=120,
                             min_occuring_temperature=0):
    """A weighted MSE metric, that adds weight to errors depending on how
    close the true temperature is to the maximum allowed temperature.
    Moreover, under-estimates are multiplied by a factor k.

    :param max_allowed_temperature: Any scalar representing the maximum
        allowed temperature.
    :param min_occuring_temperature: Any scalar representing the minimum
        temperature likely to occur in all measurements
    """
    k = np.sqrt(10)  # scale factor for under-estimates
    if hasattr(y_true, 'values'): y_true = y_true.values
    if hasattr(y_pred, 'values'): y_pred = y_pred.values

    diff = y_true - y_pred
    lower_bound = np.abs(min_occuring_temperature)
    # lift weight for estimates in critical areas
    diff *= (y_true + lower_bound) / (max_allowed_temperature +
                                              lower_bound)
    # lift weight for under-estimates
    diff[diff > 0] *= k

    return np.mean(np.average(diff ** 2, axis=0, weights=sample_weight))
