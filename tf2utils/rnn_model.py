import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tf2utils.config as cfg


class DataGenerator(Sequence):
    """When running in validation mode i.e. num profiles is less than
    batch_size, then the batch_size of an instance of this class should not be
    altered after creation."""
    def __init__(self, x, y=None, batch_size=32, tbptt_len=1,
                 increase_tbptt_factor=2, val_mode=False,
                 ):
        self.reduce_weight_on_thinner_batches = \
            cfg.keras_cfg['rnn_params']['reduce_weight_on_thinner_batches']
        self.batch_size = batch_size
        self.tbptt_len = tbptt_len
        self.increase_tbptt_factor = increase_tbptt_factor
        self.epoch_counter = 0
        self.orig_x, self.orig_y = x, y
        self.num_profiles = x.groupby(cfg.data_cfg['p_id_col']).ngroups
        self.x, self.sample_weights = self._generate_batches(self.orig_x)
        if self.orig_y is not None:
            self.y, _ = self._generate_batches(self.orig_y)

        # validation/ test set condition
        self.validation_mode = self.num_profiles < self.batch_size or val_mode
        self.val_idx = None
        if self.validation_mode:
            self.val_idx = np.tile(np.arange(self.num_profiles),
                            self.batch_size)[:self.batch_size]
            # placeholder for speed-up
            self.val_x = np.zeros([self.batch_size] + list(self.x.shape[1:]))
            if y is not None:
                self.val_y = np.zeros([self.batch_size] + list(self.y.shape[1:]))
            else:
                self.val_y = None
            self.val_sample_weights = \
                np.zeros([self.batch_size] + list(self.sample_weights.shape[1:]))

    def __getitem__(self, idx):
        """idx is the enumerated batch idx starting at 0"""
        if self.validation_mode:
            self.val_x[:] = \
                self.x[idx * self.num_profiles:
                       (idx+1) * self.num_profiles][self.val_idx]
            if self.orig_y is not None:
                self.val_y[:] = \
                    self.y[idx * self.num_profiles:
                           (idx + 1) * self.num_profiles][self.val_idx]

            x, y = self.val_x, self.val_y
        else:
            x = self.x[idx * self.batch_size: (idx+1) * self.batch_size]
            y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.sample_weights is None:
            s = None
        else:

            if self.validation_mode:
                self.val_sample_weights[:] = self.sample_weights[
                    idx * self.num_profiles:
                    (idx + 1) * self.num_profiles][self.val_idx]
                s = self.val_sample_weights
            else:
                s = self.sample_weights[
                    idx * self.batch_size:(idx + 1) * self.batch_size]
        return x, y, s

    def __len__(self):
        if self.validation_mode:
            return math.ceil(len(self.x) / self.num_profiles)
        else:
            return math.ceil(len(self.x) / self.batch_size)

    def _generate_batches(self, _df):
        """Write me"""
        p_id_col = cfg.data_cfg['p_id_col']

        grp = _df.groupby(p_id_col)
        profile_dfs_l = [df.drop(p_id_col, axis=1).reset_index(drop=True) for
                         p_id, df in grp]

        max_len = max(grp.groups.values(), key=lambda g: g.size).size

        # increase maxlen for having profiles multiples of tbptt_len.

        if max_len % self.tbptt_len > 0:
            max_len += (self.tbptt_len - (max_len % self.tbptt_len))
            max_len = int(max_len)

        # placeholder
        dummy_val = -999999
        arr = np.full((len(profile_dfs_l), max_len, _df.shape[1]-1),
                      dummy_val, dtype=np.float32)

        # give all profiles equal length where we pad with zero
        for i, profile in enumerate(profile_dfs_l):
            arr[i, :len(profile), :] = profile.to_numpy()
        arr[arr == dummy_val] = np.nan

        # break sequences along axis 1 for tbptt length
        if max_len >= self.tbptt_len:
            if max_len != self.tbptt_len:
                arr = np.vstack([arr[:, n:n + self.tbptt_len, :] for n in
                                 range(0, arr.shape[1], self.tbptt_len)])
        else:
            raise ValueError('TBPTT Len > max profile length!')
        assert arr.shape[1] % self.tbptt_len == 0, 'ping!'

        # mask the padded zeros out by giving them a sample weight of
        #  0 during training
        nan_mask = np.isnan(arr[:, :, 0].reshape(arr.shape[:2]))
        sample_weights = np.ones_like(nan_mask, dtype=np.float64)
        if np.any(nan_mask):
            sample_weights[nan_mask] = 0
            if self.reduce_weight_on_thinner_batches:
                sample_weights *= \
                    sample_weights.sum(axis=0) / sample_weights.shape[0]

        arr = np.nan_to_num(arr).astype(np.float32)

        return arr, sample_weights


class RNNKerasRegressor(KerasRegressor):
    """ScikitLearn wrapper for keras models which incorporates
    batch-generation on top. This Class wraps RNN topologies."""

    def __init__(self, *args, **kwargs):
        self.score_params = kwargs.pop('score_params', None)
        self.predict_params = kwargs.pop('predict_params', None)
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Dummy function to satisfy keras BaseWrapper"""
        pass

    def get_params(self, **params):
        res = super().get_params(**params)
        res.update({'score_params': self.score_params,
                    'predict_params': self.predict_params})
        return res

    def set_params(self, **params):
        self.predict_params = params.pop('predict_params', None)
        self.score_params = params.pop('score_params', None)
        super().set_params(**params)

    def reset_states(self):
        self.model.reset_states()

    def fit(self, x, y, **kwargs):

        assert isinstance(x, pd.DataFrame), \
            f'{self.__class__.__name__} needs pandas DataFrames as input'


        tbptt_len = kwargs.pop('tbptt_len', None)
        batch_size = kwargs.pop('batch_size', 32)

        # first conduct iteration
        batch_generation_cfg = {'batch_size': batch_size,
                                'tbptt_len': tbptt_len}

        # training set
        seq = DataGenerator(x, y, **batch_generation_cfg)

        # validation set
        x_val, y_val = kwargs.pop('validation_data', (None, None))
        if x_val is not None and y_val is not None:
            val_seq = DataGenerator(x_val, y_val, **batch_generation_cfg)
        else:
            val_seq = None

        kwargs['validation_data'] = val_seq

        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [
                EarlyStopping(monitor='val_loss',
                              min_delta=1e-3,
                              patience=cfg.keras_cfg['early_stop_patience']),
                ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=cfg.keras_cfg[
                                               'early_stop_patience'] // 3),

            ]
        if hasattr(self, 'model'):
            # already initialized
            history = self.model.fit(seq, y=None, **kwargs)
        else:
            history = super().fit(seq, y=None, **kwargs)

        return history

    def predict(self, x, **kwargs):
        """Use this func to get a prediction for x. """

        assert isinstance(x, pd.DataFrame), \
            f'{self.__class__.__name__} needs pandas DataFrames as input'

        if self.predict_params is not None:
            kwargs.update(self.predict_params.copy())

        _ = kwargs.pop('downsample_rate', 1)
        val_mode = kwargs.pop('val_mode', False)
        tbptt_len = kwargs.pop('tbptt_len', None)
        batch_size = kwargs['batch_size']

        seq = DataGenerator(x, None, batch_size, tbptt_len, val_mode=val_mode)
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        _yhat = self.model.predict(seq, **kwargs)

        def revert_reshaping(yhat, sample_w):
            if len(yhat.shape) < 3:
                if tbptt_len == 1:
                    third_dim = 1 if len(yhat.shape) == 1 else yhat.shape[1]
                    yhat = yhat.reshape(yhat.shape[0], 1, third_dim)
                elif len(yhat.shape) == 2:
                    # single target model
                    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1], 1)
                else:
                    raise ValueError('Something wrong with _yhat shape!')
            stride = seq.num_profiles if seq.validation_mode else batch_size
            if sample_w is None:
                n_dummies = stride * [0]
            else:
                n_dummies = self.get_dummies_from_w_matrix(sample_w, stride)

            # return yhat as 2-dim matrix
            # 3d due to tbptt length -> 2d

            profiles = []
            # revert breakdown due to tbptt
            for idx_b, n_dummy in enumerate(n_dummies):
                profile = np.vstack([yhat[idx_b + n, :, :] for n in
                                    range(0, yhat.shape[0], batch_size)])
                if n_dummy != 0:
                    profile = profile[:-n_dummy, :]
                profiles.append(profile)

            yhat = np.vstack(profiles)
            return yhat

        pred = revert_reshaping(np.squeeze(_yhat), seq.sample_weights)

        return pred

    @staticmethod
    def get_dummies_from_w_matrix(weights, stride):
        """Scan weight matrix for zeros which denote the padded zeros that
        need to be chopped off at the end. Return List of dummies for each
        profile which may be downsampled"""

        max_profile_len = weights.shape[0] * weights.shape[1] // stride
        n_dummies_within_batch = \
            max_profile_len - np.sum(np.count_nonzero(
                weights[n:n + stride, :], axis=1) for n in
                                     range(0, weights.shape[0], stride))
        return n_dummies_within_batch.astype(int)


class StateResetter(tf.keras.callbacks.Callback):
    """This callback helps conditioning the output states of the recurrent
    layers in an RNN architecture."""

    def __init__(self, datamanager=None, layer=None, noise=None):
        super().__init__()
        self.init_states_epoch_begin = None
        self.init_states_validation = None
        self.init_states_test = None
        if layer is not None:
            assert datamanager is not None, \
                'datamanager must be given if layer is specified'
            init_vals = pd.concat([df.iloc[0:1, :] for p, df in
                                   datamanager.df
                                  .groupby(datamanager.PROFILE_ID_COL)],
                                  ignore_index=True) \
                .set_index(datamanager.PROFILE_ID_COL)
            # DataGenerator sorts profiles during GroupBy
            #  thus, sort profile numbers here, too
            val_profiles = sorted([int(p) for p in cfg.data_cfg['valset']]) if \
                datamanager.has_hold_out else []
            train_profiles = sorted([p for p in datamanager.original_profiles if str(p)
                              not in cfg.data_cfg['testset'] +
                              [str(q) for q in val_profiles]])
            test_profiles = sorted([int(p) for p in cfg.data_cfg['testset']])
            self.init_states_epoch_begin = init_vals.loc[train_profiles, :]
            self.init_states_validation = init_vals.loc[val_profiles, :]
            self.init_states_test = init_vals.loc[test_profiles, :]
            self.init_vals = init_vals
        self.dm = datamanager
        self.num_batches = 'samples' if tf.version.VERSION.startswith('2.1.') \
            else 'steps'
        self.layer = layer  # whether to reset only a specific layer
        # if no layer is specified, init_states_.. is ignored.
        self.noise = noise

    def _reset(self, states):
        if self.layer is not None:
            if hasattr(self.init_states_epoch_begin, 'shape'):
                batch_size = self.init_states_epoch_begin.shape[0]
                if states.shape[0] < batch_size:
                    states = (np.tile(states.values, (batch_size, 1))[:batch_size, :],)
            else:  # assume init states are list like
                batch_size = self.init_states_epoch_begin[0].shape[0]
                if states[0].shape[0] < batch_size:
                    for i, s in enumerate(states):
                        states[i] = np.tile(s.values, (batch_size, 1))[:batch_size, :]
            self.model.get_layer(self.layer).reset_states(states=states)
        else:
            self.model.reset_states()

    def on_train_batch_end(self, batch, logs={}):
        # check whether we are in last batch and initialize for val set
        if batch == (self.params[self.num_batches] - 1):
            if len(self.init_states_validation) > 0:
                self._reset(self.init_states_validation)

    def on_epoch_begin(self, epoch, logs={}):
        reset_values = self.init_states_epoch_begin
        if self.noise is not None:
            # add noise with std = 0.01
            if hasattr(reset_values, 'shape'):
                reset_values += np.random.randn(*reset_values.shape)*self.noise
            else:
                # assume reset_values to be list like
                for i in range(len(reset_values)):
                    reset_values[i] += np.random.randn(*reset_values[i].shape)*0.01
        self._reset(reset_values)

    def on_train_end(self, logs={}):
        self._reset(self.init_states_test)


class IntegratorStateResetter(StateResetter):
    """Resets with initial values of the targets"""

    def __init__(self, datamanager=None, layer='rnn', noise=None):
        super().__init__(datamanager, layer, noise)
        if layer is not None:
            state_cols = cfg.data_cfg['Target_param_names']
            self.init_states_epoch_begin = \
                self.init_states_epoch_begin.loc[:, state_cols]
            self.init_states_validation = \
                self.init_states_validation.loc[:, state_cols]
            self.init_states_test = self.init_states_test.loc[:, state_cols]
            self.intermediate_init_vals_train = None

    def on_train_batch_end(self, batch, logs={}):
        last_batch = self.params[self.num_batches] - 1
        if self.intermediate_init_vals_train is not None:
            if batch < last_batch:
                self._reset(self.intermediate_init_vals_train[batch])
        super().on_train_batch_end(batch, logs)


class LPTNStateResetter(StateResetter):
    """Resets with initial target values of full dataset"""

    def __init__(self, datamanager=None, layer=None, only_these_profiles=None):

        if only_these_profiles is not None:
            assert isinstance(only_these_profiles, list), 'ping'
            self.init_vals = pd.concat([df.iloc[0:1, :] for p, df in
                                        datamanager.df
                                       .groupby(datamanager.PROFILE_ID_COL)
                                        if p in only_these_profiles],
                                       ignore_index=True) \
                .set_index(datamanager.PROFILE_ID_COL)
        else:
            super().__init__(datamanager, layer)
        if layer is not None:
            target_cols = cfg.data_cfg['Target_param_names']
            init = self.init_vals.loc[:, target_cols]
            self.init_states_epoch_begin = init
            self.init_states_validation = init
            self.init_states_test = init


class NaNCatcher(tf.keras.callbacks.Callback):
    """Stops training if NaNs occur and restores best weights."""
    NAN_CONST = float(99999)

    def __init__(self, monitor='val_loss'):
        super().__init__()
        self.monitor = monitor
        self.best = np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if np.any(np.isnan(logs.get('loss', np.NaN))):
            self._stop_training(logs)

        val_loss = logs.get('val_loss', None)
        if val_loss is not None:
            if np.any(np.isnan(val_loss)):
                self._stop_training(logs)

        if self.model.stop_training:
            # restore best weights
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
            else:
                print('Cannot restore best weights since there are no weights yet.')
        else:
            current = self.get_monitor_value(logs)
            if current is not None:
                if np.less(current, self.best):
                    self.best = current
                    self.best_weights = self.model.get_weights()

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print('Early stopping conditioned on metric `%s` '
                  'which is not available. Available metrics are: %s',
                  self.monitor, ','.join(list(logs.keys())))
        return monitor_value

    def _stop_training(self, _logs):
        self.model.stop_training = True
        _logs['loss'] = self.NAN_CONST
        if 'val_loss' in _logs:
            _logs['val_loss'] = self.NAN_CONST
        print('Stop training due to nan output')
        self.model.history.history['nan_output'] = True
