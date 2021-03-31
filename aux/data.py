"""
Author: Kirgsn, 2018, https://www.kaggle.com/wkirgsn
"""
from abc import ABC, abstractmethod
import gc
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
    PolynomialFeatures, QuantileTransformer, RobustScaler
from sklearn.metrics import mean_squared_error as mse, mean_squared_log_error\
    as msle, mean_absolute_error as mae, r2_score
from tensorflow.keras.losses import mean_squared_error, mean_squared_logarithmic_error

import aux.config as cfg
from aux.general_model import SimpleScaler, OnMaxAbsValueScaler


class DataManagerScorer:
    """For sklearn functions that requirer a scorer. With this class we can
    un-standardize according to our datamanager and get the unnormalized
    score"""
    __name__ = 'data_manager_loss'  # for tpot
    losses = {'mse': mse, 'msle': msle, 'mae': mae}

    def __init__(self, datamanager, force_numpy=True, force_ycol=None,
                 loss='mse'):
        self.dm = datamanager
        self.force_numpy = force_numpy
        self.force_ycol = force_ycol
        self.loss_func = self.losses[loss]

    def __call__(self, estimator, X, y):
        """Score function"""
        try:
            score, _, _ = self.score(estimator, X, y)
        except ValueError:
            # nans
            score = 9999
        return score

    def score(self, estimator, X, y):
        if self.force_ycol is not None:
            y_col = self.force_ycol
            y_test = y
        elif isinstance(y, pd.Series):
            y_col = [y.name]
            y_test = y.values
        elif isinstance(y, pd.DataFrame):
            y_col = [col for col in y.columns.tolist() if
                     col != self.dm.PROFILE_ID_COL]
            y_test = y.loc[:, y_col].values
        else:
            # user hasnt specified target and y_test is numpy matrix so
            # assume targets from config file
            y_col = cfg.data_cfg['Target_param_names']
            y_test = y

        y_hat = estimator.predict(X)
        y_hat = self.dm.inverse_transform(y_hat, cols_to_inverse=y_col)
        y_test = self.dm.inverse_transform(y_test, cols_to_inverse=y_col)
        score = self.loss_func(y_test, y_hat)
        if self.force_numpy and hasattr(score, 'values'):
            score = score.values
        return score, y_test, y_hat


class DataManager(ABC):
    """Abstract dataset managing class"""

    START_OF_PROFILE_COL = 'p_start'
    loss_funcs = {'mse': mean_squared_error,
                  'msle': mean_squared_logarithmic_error}

    def __init__(self, path, create_holdout=True, alternative_pid=None):
        self.PROFILE_ID_COL = cfg.data_cfg['p_id_col']
        conversion_table = {col: np.float32 for col in cfg.data_cfg[
            'Input_param_names']+cfg.data_cfg['Target_param_names']}
        conversion_table.update({self.PROFILE_ID_COL: np.uint8})
        # original data
        self.dataset = pd.read_csv(path, dtype=conversion_table)
        # When using CV, do not create a hold out
        self.has_hold_out = create_holdout
        self.loss_func = self.loss_funcs[cfg.data_cfg['loss']]

        if alternative_pid is not None:
            if self.PROFILE_ID_COL in self.dataset:
                self.dataset = self.dataset.drop([self.PROFILE_ID_COL], axis=1)
            self.dataset = self.dataset.rename(
                columns={alternative_pid: self.PROFILE_ID_COL})

        # Drop profiles
        if cfg.data_cfg['drop'] is not None:
            assert isinstance(cfg.data_cfg['drop'], list), \
                'please specify profiles to drop as list'
            drop_p_l = [int(p) for p in cfg.data_cfg['drop']]
            self.dataset = self.dataset.query(
                f'{self.PROFILE_ID_COL} not in @drop_p_l') \
                .reset_index(drop=True)

        # downsample
        self.downsample_rate = cfg.data_cfg['downsample_rate']
        if self.downsample_rate > 1:
            sample_time_in_ms = int(cfg.data_cfg['sample_time'][:-2])
            new_sample_time_in_ms = sample_time_in_ms * self.downsample_rate

            self.dataset = pd.concat(
                [df.assign(time=pd.date_range(start=0, periods=len(df),
                                              freq=cfg.data_cfg['sample_time']))
                   .resample(f'{new_sample_time_in_ms}ms', on='time').mean()
                   .reset_index(drop=True)
                 for pid, df in
                 self.dataset.groupby(self.PROFILE_ID_COL)], ignore_index=True)

        self._x_cols, self._y_cols = cfg.data_cfg['Input_param_names'], \
                                     cfg.data_cfg['Target_param_names']
        self.original_profiles = self.dataset[self.PROFILE_ID_COL].unique()\
            .tolist()

        # drop columns
        relevant_columns = list(set(self._x_cols) | set(self._y_cols)) + \
                           [self.PROFILE_ID_COL]
        self.dataset = self.dataset.loc[:, relevant_columns]
        if cfg.debug_cfg['DEBUG']:
            self.dataset = pd.concat(
                [df.iloc[:cfg.debug_cfg['n_debug'], :] for
                 p_id, df in self.dataset.groupby(self.PROFILE_ID_COL)],
                ignore_index=True
            )
        else:
            # sort dataset such that profiles appear in increasing p_id order
            # groupby does sorting implicitly

            self.dataset = \
                pd.concat([df for _, df in
                           self.dataset.groupby([self.PROFILE_ID_COL])],
                          ignore_index=True)

        self.df = self.dataset.copy()

    @property
    def x_cols(self):
        return self._x_cols

    @x_cols.setter
    def x_cols(self, value):
        self._x_cols = value

    @property
    def y_cols(self):
        return self._y_cols

    @y_cols.setter
    def y_cols(self, value):
        self._y_cols = value

    @property
    def profiles_df(self):
        """list of dfs per profile"""
        return [df.reset_index(drop=True) for _, df in
                self.df.groupby([self.PROFILE_ID_COL])]

    @property
    def tra_df(self):
        testsets = [int(x) for x in cfg.data_cfg['testset']]
        valsets = [int(x) for x in cfg.data_cfg['valset']]
        profiles_to_exclude = \
            testsets + valsets if self.has_hold_out else testsets

        return self.df\
            .query(f'{self.PROFILE_ID_COL} not in @profiles_to_exclude')\
            .reset_index(drop=True)

    @property
    def val_df(self):
        valset = [int(x) for x in cfg.data_cfg['valset']]
        sub_df = self.df.loc[self.df[self.PROFILE_ID_COL].isin(valset), :]
        return sub_df.reset_index(drop=True)

    @property
    def tst_df(self):
        testset = [int(x) for x in cfg.data_cfg['testset']]
        sub_df = self.df.loc[self.df[self.PROFILE_ID_COL].isin(testset), :]
        return sub_df.reset_index(drop=True)

    @property
    def tst_orig_targets(self):
        testset = [int(x) for x in cfg.data_cfg['testset']]
        return self.dataset.loc[self.dataset[self.PROFILE_ID_COL].isin(testset),
                                self.y_cols].reset_index(drop=True)

    @property
    def actual(self):
        testset = [int(x) for x in cfg.data_cfg['testset']]
        sub_df = \
            self.dataset.loc[
            self.dataset[self.PROFILE_ID_COL].isin(testset),
            cfg.data_cfg['Target_param_names'] + [self.PROFILE_ID_COL]]
        return sub_df.reset_index(drop=True)

    @abstractmethod
    def get_featurized_sets(self):
        pass

    @abstractmethod
    def inverse_transform(self, _yhat):
        pass

    def score(self, y_hat, y_true=None, verbose=True):
        """Prints score by comparing given y_hat with dataset's target,
        which is in the testset. Returns the actual target data as well"""
        nan_score = float(9999)
        act = y_true if y_true is not None else self.actual[self.y_cols]
        if hasattr(act, 'values'):
            act = act.values
        if hasattr(y_hat, 'values'):
            y_hat = y_hat.values
        assert act.shape == y_hat.shape, f'shape mismatch in DM.score(): ' \
                                         f'{act.shape} != {y_hat.shape}'
        sklearn_metrics = {'mse': mse, 'msle': msle}
        lossf = sklearn_metrics[cfg.data_cfg['loss']]
        try:
            score = lossf(act, y_hat)
        except ValueError as err:
            print(err)
            print(f'NaNs in prediction. Setting score to {nan_score}.')
            score = nan_score
        return score, act

    def get_p_id(self, _df):
        """Get Profile ID of given dataframe. Raises error, if there are
        more than one profile id"""
        p_ids = _df[self.PROFILE_ID_COL].unique().tolist()
        assert len(p_ids) == 1, 'More than one profile given in get_p_id()!'
        return p_ids[0]

    def reset(self):
        del self.df
        self.df = self.dataset.copy()
        self._x_cols, self._y_cols = cfg.data_cfg['Input_param_names'], \
                                     cfg.data_cfg['Target_param_names']
        self.original_profiles = self.dataset[self.PROFILE_ID_COL].unique() \
            .tolist()


class LightDataManager(DataManager):
    """Lightweight data managing without scikit pipelines"""

    def __init__(self, path, has_holdout=True, create_rolling_diffs=False,
                 standardize=True, alternative_pid=None,
                 estimate_diffs=False):
        super().__init__(path=path, create_holdout=has_holdout,
                         alternative_pid=alternative_pid)
        self.rolling_lookback = cfg.data_cfg['rolling_lookback']
        self.standardize = standardize
        self.create_rolling_diffs = create_rolling_diffs
        self.estimate_diffs = estimate_diffs
        self._reset_scaler()

    def reset(self):
        super().reset()
        # for benchmark learn curves important
        self.rolling_lookback = cfg.data_cfg['rolling_lookback']
        self._reset_scaler()

    def _reset_scaler(self):

        self.scaler = StandardScaler()
        # Below are necessary only for standardize = 'simple'
        temperature_scale = cfg.keras_cfg['tnn_params']['temp_scale']
        self.scaler_temps = SimpleScaler(scale=temperature_scale)
        self.scaler_currents = SimpleScaler(scale=200)
        self.scaler_rest = OnMaxAbsValueScaler()

    def get_scorer(self, force_numpy=True):
        return DataManagerScorer(self, force_numpy)

    def inverse_transform(self, df, cols_to_inverse=None):
        """If cols_to_inverse == None then assume target columns. Always
        returns a pandas DataFrame"""

        # drop p_id column
        profile_ids = None
        if isinstance(df, pd.DataFrame):
            if self.PROFILE_ID_COL in df:
                profile_ids = df.pop(self.PROFILE_ID_COL)
            arr = df.to_numpy()
        else:
            arr = df
        if profile_ids is None:
            profile_ids = self.tst_df.loc[:, self.PROFILE_ID_COL]

        if len(arr.shape) == 1:  # make at least 2 dim
            arr = arr.reshape([-1, 1])

        if cols_to_inverse is None:
            # assumption: df contains targets only
            assert arr.shape[1] == len(self.y_cols), \
                f'target mismatch {arr.shape[1]} != {len(self.y_cols)}'
            cols_to_inverse = self.y_cols

        # inverse scaling
        if isinstance(self.standardize, bool):
            if self.standardize:
                if cols_to_inverse == self.y_cols and len(self.y_cols) == 4:
                    # extra scaling of targets
                    arr *= self.target_stds[2] / self.target_stds

                orig_scaling_cols = [x for x in self.df.columns if '_bin_' not in
                                     x and x != self.PROFILE_ID_COL
                                     and x != 'time']
                inversed = pd.DataFrame(np.zeros((len(df), len(orig_scaling_cols))),
                                        columns=orig_scaling_cols)
                inversed.loc[:, cols_to_inverse] = arr
                inversed.loc[:, orig_scaling_cols] = \
                    self.scaler.inverse_transform(inversed)
                inversed = inversed.loc[:, cols_to_inverse]
            else:
                if isinstance(df, pd.DataFrame):
                    inversed = df
                else:
                    inversed = pd.DataFrame(arr, columns=cols_to_inverse)
        else:
            assert isinstance(self.standardize, str), \
                f'{self.standardize} is neither bool nor str'
            if self.standardize == 'simple':
                temperature_cols = [c for c in cfg.data_cfg['temperature_cols']
                                    if c in self.x_cols + self.y_cols]
                n_left_pad = len(temperature_cols) - len(self.y_cols)
                inversed = self.scaler_temps.inverse_transform(
                    np.hstack([np.zeros((len(arr), n_left_pad)), arr])
                )[:, -len(self.y_cols):]
                inversed = pd.DataFrame(inversed, columns=cols_to_inverse)
            else:
                raise ValueError(f'{self.standardize} not allowed.')

        if self.estimate_diffs:
            assert len(inversed) == len(profile_ids), \
                f'length mismatch {len(inversed)} != {len(profile_ids)}'

            inversed[self.PROFILE_ID_COL] = profile_ids

            def invert_shift(p_df, p_init_vals):
                p_df = p_df.drop([self.PROFILE_ID_COL], axis=1)
                p_df.iloc[0, :] = p_init_vals
                return p_df.cumsum()

            inversed = pd.concat([invert_shift(_df, self.initial_y.loc[p_id, :])
                                  for p_id, _df in
                                  inversed.groupby(self.PROFILE_ID_COL)])
        return inversed

    def _update_x_cols(self):
        self.x_cols = [x for x in self.df.columns.tolist() if x not
                       in cfg.data_cfg['Target_param_names'] +
                       [self.PROFILE_ID_COL]]

    def featurize(self, jobs=4, create_polynomials=False, create_ewmas=True,
                  create_lag_feats=False, y_smoothing=None, drop_dq=False):
        """Conduct feature engineering"""
        print('build dataset ..')
        # extra features
        if {'i_d', 'i_q', 'u_d', 'u_q'}.issubset(
                set(self.df.columns.tolist())):
            extra_feats = \
                {'i_s': lambda x: np.sqrt((x['i_d']**2 + x['i_q']**2) ),
                 'u_s': lambda x: np.sqrt((x['u_d']**2 + x['u_q']**2)),
                 #'S_el': lambda x: x['i_s']*x['u_s'],
                 #'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],
                 #'S_el+coolant': lambda x: x['S_el'] + x['coolant'],
                 #'triple': lambda x: x['S_el+coolant'] + x['motor_speed'],
                 #'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
                 #'S_x_w': lambda x: x['S_el']*x['motor_speed'],
                 #'Psi_d_dot': lambda x: x['u_d'] - x['i_d'] -
                 #                       x['motor_speed']*x['i_q'],
                 #'Psi_q_dot': lambda x: x['u_q'] + x['i_q'] -
                 #                       x['motor_speed'] +
                 #                       x['motor_speed']*x['i_d'],
                 #'Psi_q': lambda x: x['Psi_q_dot'].cumsum(),
                 #'Psi_d': lambda x: x['Psi_d_dot'].cumsum(),
                 #'P_mech': lambda x: 2 * np.pi * x['motor_speed']
                 #                    * x['torque'] / 60,
                 #'d_P': lambda x: abs(x['P_el'] - x['P_mech'])
                 }

            self.df = self.df.assign(**extra_feats)
            if drop_dq:
                self.df = self.df.drop([c for c in self.df if
                                        c.endswith(('_q', '_d'))], axis=1)
            self.df.columns = [c.replace(' ', '_') for c in self.df.columns]

        self._update_x_cols()

        # engineer y cols
        if y_smoothing is not None:
            orig_target = self.df.loc[:, self.y_cols]
            y_smoothed = [y.rolling(y_smoothing,
                                    center=True).mean() for p_id, y in
                          self.df[self.y_cols + [self.PROFILE_ID_COL]]
                              .groupby(self.PROFILE_ID_COL)]
            self.df.loc[:, self.y_cols] = \
                pd.concat(y_smoothed).fillna(orig_target)

        # engineer x cols
        cols_to_smooth = ['ambient', 'coolant']
        smoothing_window = 100
        if all(c in self.df for c in cols_to_smooth):
            orig_x = self.df.loc[:, cols_to_smooth]
            x_smoothed = [x.rolling(smoothing_window,
                                    center=True).mean() for p_id, x in
                          self.df[cols_to_smooth + [self.PROFILE_ID_COL]]
                              .groupby(self.PROFILE_ID_COL)]
            self.df.loc[:, cols_to_smooth] = \
                pd.concat(x_smoothed).fillna(orig_x)

        p_df_list = [df.drop(self.PROFILE_ID_COL, axis=1).reset_index(drop=True)
                     for _, df in
                     self.df[self.x_cols + [self.PROFILE_ID_COL]]
                         .groupby([self.PROFILE_ID_COL])]

        to_merge = [self.df]

        if create_ewmas or create_lag_feats:
            with multiprocessing.Pool(jobs) as pool:
                def apply_parallel(func):
                    ret = pool.map(func, p_df_list)

                    to_merge.append(pd.concat(ret, ignore_index=True))
                    return ret

                if create_lag_feats:
                    apply_parallel(self._dig_into_lag_features)
                if create_ewmas:
                    apply_parallel(self._dig_into_rolling_features)

        # merge features together and drop NAN
        self.df = pd.concat(to_merge, axis=1).dropna().reset_index(drop=True)
        self._update_x_cols()

        # polynomials
        if create_polynomials:
            preserved_df = self.df.loc[:, self.y_cols + [self.PROFILE_ID_COL]]
            poly = PolynomialFeatures()
            polynomials_arr = poly.fit_transform(self.df[self.x_cols])
            self.x_cols = poly.get_feature_names(input_features=self.x_cols)
            poly_df = pd.DataFrame(polynomials_arr, columns=self.x_cols)
            self.df = pd.concat([poly_df, preserved_df], axis=1)

        # estimate diffs instead of absolute values
        if self.estimate_diffs:
            grp_y = self.df.loc[:, self.y_cols + [self.PROFILE_ID_COL]]\
                        .groupby(self.PROFILE_ID_COL)
            self.initial_y = pd.concat([y.iloc[0:1, :] for p_id, y in
                                       grp_y]).set_index(self.PROFILE_ID_COL)

            y_diff = pd.concat([y - y.shift(1).fillna(y.iloc[0, :])
                                for pid, y in grp_y])
            self.df.loc[:, self.y_cols] = y_diff.loc[:, self.y_cols]

        # standardize
        if isinstance(self.standardize, bool):
            if self.standardize:
                # extra scaling in targets
                self.target_stds = self.tra_df[self.y_cols].std()
                self.scaler.fit(self.tra_df.drop(self.PROFILE_ID_COL,
                                                 axis=1).astype(np.float32))
                p_ids = self.df.pop(self.PROFILE_ID_COL)
                self.df = pd.DataFrame(self.scaler.transform(self.df),
                                       columns=self.df.columns,
                                       dtype=np.float32)
                self.df[self.PROFILE_ID_COL] = p_ids
                # extra scaling in targets
                if len(self.y_cols) == 4:
                    self.df.loc[:, self.y_cols] *= self.target_stds / self.target_stds[2]
        else:
            assert isinstance(self.standardize, str), \
                f'{self.standardize} is neither bool nor str'
            if self.standardize == 'simple':
                # special normalization (minmax, feature range dependent on
                # specific feature).
                #  This shall help to preserve domain knowledge
                temperature_cols = [c for c in cfg.data_cfg['temperature_cols']
                                    if c in self.x_cols + self.y_cols]
                current_cols = [c for c in self.x_cols if c.startswith('i_')]
                rest_cols = [c for c in self.x_cols
                             if c not in temperature_cols + current_cols]
                # scale
                self.df.loc[:, temperature_cols] = \
                    self.scaler_temps.fit_transform(self.df.loc[:, temperature_cols])
                self.df.loc[:, current_cols] = \
                    self.scaler_currents.fit_transform(self.df.loc[:, current_cols])
                self.df.loc[:, rest_cols] = \
                    self.scaler_rest.fit_transform(self.df.loc[:, rest_cols])
            else:
                raise ValueError(f'{self.standardize} not allowed. '
                                 f'Must be in [simple]')

        gc.collect()
        print(f'{self.df.memory_usage().sum() / 1024**2:.2f} MB with '
              f'{len(self.x_cols)} input features')
        return self

    def _dig_into_lag_features(self, df):
        # Here, lookback is always equal to last seen observation (lagX)
        lookback = 8
        n_lookback = 4
        dfs = []
        for lback in range(lookback, n_lookback*lookback + 1, lookback):
            # e.g. lback € [32,64,96,128], for lookback=32 and n_lookback=4
            lag_feats = [df.shift(lback).astype(np.float32)
                           .fillna(df.iloc[0, :])
                           .add_suffix(f'_lag_{lback}'),
                         #df.diff(periods=lback).astype(np.float32)
                         #  .fillna(df.iloc[0, :])
                         #  .add_suffix(f'_lag_{lback}_diff')
                         ]

            """lag_feats += [abs(lag_feats[1]).astype(np.float32)
                              .add_suffix('_abs'),
                          pd.DataFrame(df.values + lag_feats[0].values,
                                       columns=df.columns)
                              .add_suffix(f'_sum')]"""

            dfs.append(pd.concat(lag_feats, axis=1))
        ret = pd.concat(dfs, axis=1)
        return ret

    def _dig_into_rolling_features(self, df):
        lookback = self.rolling_lookback
        if not isinstance(lookback, list):
            lookback = [lookback]

        # get max lookback
        max_lookback = max(lookback)
        # prepad default values until max lookback in order to get unbiased
        # rolling lookback feature during first observations
        dummy = pd.DataFrame(np.zeros((max_lookback, len(df.columns))),
                             columns=df.columns)

        temperature_cols = [c for c in ['ambient', 'coolant'] if c in df]
        dummy.loc[:, temperature_cols] = df.loc[0, temperature_cols].values

        # prepad
        df = pd.concat([dummy, df], axis=0, ignore_index=True)

        ew_mean = [df.ewm(span=lb).mean()
                       .rename(columns=lambda c: c+'_ewma_'+str(lb))
                   for lb in lookback]
        ew_mean_diff = [-(mu - df.values).rename(columns=lambda c: c+'_diff')
                        for mu in ew_mean]
        ew_std = pd.concat(
            [df.ewm(span=lb).std().astype(np.float32)
                 .rename(columns=lambda c: c+'_ewms_'+str(lb))
             for lb in lookback], axis=1)

        concat_l = [pd.concat(ew_mean, axis=1).astype(np.float32),
                    ew_std.fillna(0),
                    #pd.concat(ew_mean_diff, axis=1).astype(np.float32)
                    #self._calc_quantiles(df),
                    ]
        ret = pd.concat(concat_l, axis=1).iloc[max_lookback:, :]\
            .reset_index(drop=True)

        if self.create_rolling_diffs:
            diff_d = {}
            for i in range(1, len(lookback)):
                lb = lookback[i]
                lb_prev = lookback[i-1]
                diff_d.update(
                    {f'{c.split("_ew_rolling")[0]}_ew_rolling_mean_diff_{lb}'
                     f'_{lb_prev}':
                         lambda x: x[c] - x[f'{c.split("_ew_rolling")[0]}'
                         f'_ew_rolling_mean_{lb_prev}'] for c in
                     ew_mean.columns if c.endswith(str(lb))})
            ret = ret.assign(**diff_d)
        return ret

    @staticmethod
    def _calc_quantiles(_df):
        def list_flatten(l):
            return [item for sublist in l for item in sublist]
        quantiles = [0.05,
                      .1,
                     .2,
                     # .4, .6,
                     .8,
                     .9,
                     .95]
        lbs = [500, 1500, 2500]
        rolling_objs = [_df.rolling(lb) for lb in lbs]

        rolling_qs_l = []
        for i, rol in enumerate(rolling_objs):
            rolling_qs_l.append({q: rol.quantile(q).fillna(0).rename(
                columns=lambda c: c+f'_{lbs[i]}_lb_') for q in
                quantiles})

        q_l = [[(q_d[.95] - q_d[.05])
                        .rename(columns=lambda c: f'{c}_quants_5perc_diff'),
               (q_d[.9] - q_d[.1])
                   .rename(columns=lambda c: f'{c}_quants_10perc_diff'),
                (q_d[.8] - q_d[.2])
                    .rename(columns=lambda c: f'{c}_quants_20perc_diff'),
                # quantiles_d[.6] - quantiles_d[.4],
                ] for q_d in rolling_qs_l]
        return pd.concat(list_flatten(q_l), axis=1)

    def _get_noise_augmented_features(self, n_enrich=None):
        if n_enrich is None:
            # calculate required n_enrich
            batchsize = cfg.keras_cfg['rnn_params']['batch_size']
            n_enrich = batchsize // self.downsample_rate - 1

        noisy_dfs = []
        col_filter = [col for col in self.x_cols if 'profile' not in col]
        for n in range(1, n_enrich+1):
            np.random.seed(n)
            # todo: find appropriate noise std per feature
            dummy_noise_level = \
                np.array([1e-4]*len(col_filter))

            df = self.df.copy()
            tmp = self.df.loc[:, col_filter]
            df.loc[:, col_filter] = \
                tmp * (1 + dummy_noise_level * np.random.randn(*tmp.shape))
            df.loc[:, self.PROFILE_ID_COL] += (100*n)
            noisy_dfs.append(df)
        np.random.seed(cfg.data_cfg['random_seed'])
        return noisy_dfs

    def __get_noise_augmented_features(self):
        pass

    def get_featurized_sets(self):
        return self.tra_df, self.val_df, self.tst_df

