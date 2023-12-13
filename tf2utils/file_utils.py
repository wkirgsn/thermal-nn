import os
import time
import uuid
import random
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
import tensorflow as tf

from tf2utils import config as cfg
from tf2utils.data import LightDataManager
from tf2utils.metrics import print_scores


def measure_time(func):
    """time measuring decorator"""
    def wrapped(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('took {:.3} seconds'.format(end_time-start_time))
        return ret
    return wrapped


class catchtime:
    """time measuring context manager"""
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.e = time.time()

    def __float__(self):
        return float(self.e - self.t)

    def __coerce__(self, other):
        return (float(self), other)


class Report:
    """Summary of an experiment/trial"""

    param_map = {'pm': '{PM}',
                 'stator_tooth': '{ST}',
                 'stator_yoke': '{SY}',
                 'stator_winding': '{SW}',
                 'motor_speed': 'Motor speed',
                 'ambient': 'Ambient temperature',
                 'coolant': 'Coolant temperature',
                 'torque': 'Torque'}
    output_param_map = {'pm': 'magnet temperature',
                        'stator_tooth': 'stator tooth temperature',
                        'stator_yoke': 'stator yoke temperature',
                        'stator_winding': 'stator winding temperature'}

    time_format = "%Y-%m-%d %H:%M"

    def __init__(self, uid=None, seed=None,
                 score=None, yhat=None, actual=None, history=None,
                 used_loss=None, model=None, mad=None, db_path=None,
                 profile_ids=None, scriptname='', dm=None):
        self.score = score
        self.mad = mad
        self.yhat_te = yhat
        self.actual = actual
        self.history = history
        self.uid = uid or str(uuid.uuid4())
        self.yhat_tr = None
        self.start_time = datetime.utcnow().strftime(self.time_format)
        self.used_loss = used_loss
        self.model = model
        self.cfg_blob = {}
        self.db_path = db_path or cfg.data_cfg['db_path']
        self.predicted_profiles = profile_ids or cfg.data_cfg['testset']
        self.scriptname = scriptname
        self.seed = seed or np.random.randint(0, int(1e7))
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        self.metrics = None
        self.dm = dm
        clr_names = ['turquoise', 'yellow', 'violet',
                 'red', 'blue', 'orange', 'green', 'grey']
        clr_sets =\
            {'dark_background': {k: v for k, v in zip(
                 clr_names + ['other_{}'.format(i) for i in range(5)],
                 [rgb2hex(c[:3]) for c in plt.cm.Set3(np.linspace(0, 1, 12))])}
             }
        self.clrs = clr_sets.get(cfg.plot_cfg['style'],
                                 {k: f'tab:{k}' for k in clr_names})
        self.node = platform.uname().node

    def _format_plot(self, y_lbl='temp', x_lbl=True, legend=True,
                     legend_loc='best', ax=None):

        ax = ax or plt.gca()
        if x_lbl:
            ax.set_xlabel('Time in h')

        if y_lbl in ['temp', 'coolant', 'ambient', 'pm', 'stator_yoke',
                     'stator_teeth', 'stator_winding']:
            ax.set_ylabel('Temp. in °C')
        elif y_lbl == 'motor_speed':
            ax.set_ylabel('Motor speed\nin 1/min')
        elif y_lbl == 'torque':
            ax.set_ylabel('Torque in Nm')
        elif y_lbl.startswith('i_'):
            ax.set_ylabel('Current\n in A')

        if legend:
            ax.legend(loc=legend_loc)
        ax.set_xlim(-1000, np.around(len(self.actual), -3) + 300)
        tcks = np.arange(0, np.around(len(self.actual), -3), 7200)
        tcks_lbls = tcks // 7200 if x_lbl else []
        ax.set_xticks(tcks)
        ax.set_xticklabels(tcks_lbls)

    def plot_history(self):
        if self.history is not None:
            history = self.history.history
            plt.figure(figsize=(6, 4))
            if 'loss' in history:
                plt.plot(history['loss'], label='train loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='validation loss')
            plt.xlabel('epoch')
            plt.ylabel(f'{self.used_loss} in K²')
            plt.title(f'Training/Validation Score over Epochs of Experiment '
                      f'{self.uid}')
            plt.semilogy()
            plt.legend()

    def _trunc_pred_and_actual(self, first_loc=3, trunc_last=False):
        # 19.07.2019: HACK: Cut first second, too, to avoid crazy
        #  large deviation which happens there especially for CNNs

        self.scnd_trunc_loc += first_loc
        self.yhat_te = pd.concat([self.yhat_te.iloc[first_loc:self.first_trunc_loc, :],
                                  self.yhat_te.iloc[self.scnd_trunc_loc:]])\
            .reset_index(drop=True)
        self.actual = pd.concat([self.actual.iloc[first_loc:self.first_trunc_loc, :],
                                  self.actual.iloc[self.scnd_trunc_loc:]])\
            .reset_index(drop=True)
        self.first_trunc_loc -= first_loc
        if trunc_last:
            self.yhat_te = self.yhat_te.iloc[:self.scnd_trunc_loc, :].reset_index(drop=True)
            self.actual = self.actual.iloc[:self.scnd_trunc_loc, :].reset_index(drop=True)

    def plot(self, show=True, with_input=False, with_hist=False):
        os.makedirs('data/plots', exist_ok=True)
        plt.style.use(cfg.plot_cfg['style'])
        self.plot_history()
        self.plot_compact_testset_error(with_input, with_hist)
        self.plot_residual_over_y_range()
        if with_hist:
            try:
                self.plot_residual_histogram()
                pass
            except Exception as err:
                print(err)
                print('cant plot residual plot (histogram)')

        if show:
            plt.show()

    def plot_compact_testset_error(self, with_input=True, with_hist=False):
        n_targets = len(self.actual.columns)
        datapath = 'data/input/resting_measures.csv' \
            if 'resting' in self.scriptname else cfg.data_cfg['file_path']
        if 'resting' in self.scriptname:
            cfg.data_cfg['downsample_rate'] = 1
        if self.dm is None:
            if 'torque' not in cfg.data_cfg['Input_param_names']:
                cfg.data_cfg['Input_param_names'] += ['torque']
            cfg.data_cfg['testset'] = [str(i) for i in self.predicted_profiles]
            self.dm = LightDataManager(datapath, standardize=False,
                                       alternative_pid='rest_no' if 'resting'
                                                in self.scriptname else None)
        if isinstance(with_input, bool):
            rows = 3 if with_input else 2
            tst_df = None
            input_cols = ['motor_speed', 'torque', 'ambient', 'coolant']
        elif isinstance(with_input, (np.ndarray, pd.DataFrame)):
            rows = 3
            tst_df = with_input
            with_input = True
            input_cols = ['motor_speed', 'i_s', 'ambient', 'coolant']
        plot_length = 1.5 * rows

        diff = self.yhat_te - self.actual
        profile_lens = {pid: len(df_) for pid, df_ in
                        self.dm.dataset.groupby(self.dm.PROFILE_ID_COL,
                                                sort=False)}
        vlines_x = np.cumsum(np.array([profile_lens[int(pid)] for pid in
                                       sorted(self.predicted_profiles)]))[:-1]
        min_y_abs = min(self.yhat_te.min().min(), self.actual.min().min())
        max_y_abs = 150  #max(self.yhat_te.min().max(), self.actual.max().max())
        min_y_err = diff.min().min()
        max_y_err = diff.max().max()

        annot_bbox_kws = {'facecolor': 'white', 'edgecolor': 'black',
                          'alpha': 0.3, 'pad': 1.0}
        vlines_kws = dict(colors='k', ls='dashed', zorder=3)

        fig, axes = plt.subplots(rows, n_targets, sharex=True, sharey='row',
                                 figsize=(10/4 * n_targets, plot_length))
        for i, c in enumerate(self.actual):
            # plot signal measured and estimated
            # todo: Having only 1 target will break here
            #  axes is 1d then
            ax = axes[0, i]
            ax.set_title(r'$\vartheta_\mathregular{}$'.format(self.param_map.get(c, 'x')),
                         fontdict=dict(fontsize=12))
            ax.plot(self.actual[c], color='lime',
                    label='Ground truth', linestyle='-')
            ax.plot(self.yhat_te[c], color='xkcd:indigo',
                    label='Estimate', linestyle='-')
            ax.set_xlim(-1000, np.around(len(self.actual), -3) + 300)
            ax.vlines(vlines_x, min_y_abs, max_y_abs, **vlines_kws)
            tcks = np.arange(0, np.around(len(self.actual), -3), 7200)
            tcks_lbls = tcks // 7200
            if i == 0:
                ax.set_ylabel('Temperature in °C')
                """handles=[mpatches.Patch(color=i.get_color(),
                                                        lw=0.5,
                                                        label=i.get_label())
                                         for i in ax.get_lines()],"""
                ax.legend(loc='upper left', frameon=True, ncol=2,
                          framealpha=1.0, mode='expand')

            ax.set_xticks(tcks)
            ax.set_xticklabels(tcks_lbls)
            ax.set_ylim(None, 151)
            ax.grid(alpha=0.5)

            # plot signal estimation error
            ax = axes[1, i]
            ax.plot(diff[c], color='crimson',
                    label='Temperature Estimation error ' +
                          r'$\vartheta_{}$'.format(self.param_map.get(c, 'x')))
            ax.vlines(vlines_x, min_y_err, max_y_err, **vlines_kws)
            if i == 0:
                ax.set_ylabel('Error in °C')

            ax.text(0.5, 1.03,
                    s=f'MSE: {(diff[c] ** 2).mean():.2f} (°C)², ' +
                      r'$||e||_\infty$: ' + f'{diff[c].abs().max():.2f} °C',
                    bbox=annot_bbox_kws,
                    transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='center')
            ax.grid(alpha=0.5)
            if not with_input:
                ax.set_xlabel('Time in hours')
        if with_input:

            assert len(input_cols) == 4, 'other than 4 inputs not implemented'
            assert n_targets <= 4, 'more than 4 targets not implemented'
            input_cols = input_cols[:n_targets]

            if tst_df is None:
                cfg.data_cfg['subsequences'] = False
                self.dm.featurize(create_ewmas=False)
                tst_df = \
                    self.dm.df.query('profile_id in '
                                     '@self.predicted_profiles')\
                              .loc[:, input_cols]\
                              .reset_index(drop=True)
            # normalize
            tst_df = (tst_df - tst_df.min(axis=0)) / (tst_df.max(axis=0) -
                                                      tst_df.min(axis=0))
            legend_locs = {'motor_speed': (0.36, 0.7),
                           'torque': (0.4, 0.7),
                           'ambient': 'lower center',
                           'coolant': (0.4, 0.7)}
            input_ylbl_d = {'motor_speed': 'Motor speed\nin 1/min',
                            'torque': 'Torque in Nm',
                            'i_d': 'Current\n in A',
                            'i_q': 'Current\n in A',
                            'coolant': 'Coolant temp. in °C',
                            'ambient': 'Ambient temp. in °C'}
            for j, col in enumerate(input_cols):
                ax = axes[2, j]
                ax.plot(tst_df[col], color='tab:grey',
                        label=self.param_map.get(col, col))
                ax.set_xlabel('Time in hours')
                ax.set_title(self.param_map.get(col, col))
                if j == 0:
                    ax.set_ylabel('Normalized qty')
                ax.grid(alpha=0.5)
                ax.vlines(vlines_x, 0, 1, **vlines_kws)
        fig.tight_layout()
        return fig

    def plot_residual_histogram(self):
        for c in self.actual:
            diff = self.yhat_te[c] - self.actual[c]
            diff = np.clip(diff, a_min=-10, a_max=10)
            plt.figure(figsize=(5, 3))
            sns.distplot(diff)
            plt.xlabel(c + ' error in °C')
            plt.tight_layout()
            plt.title('Residual histogram')

    def plot_residual_over_y_range(self):
        n_targets = len(self.actual.columns)
        rows = 1
        plot_length = 2 * rows
        fig, axes = plt.subplots(rows, n_targets, sharex=True, sharey='row',
                                 figsize=(n_targets*10/4, plot_length))
        for i, (c, ax) in enumerate(zip(self.actual, axes.flatten())):
            # plot signal measured and estimated
            residuals = \
                (pd.DataFrame({c + '_true': self.actual[c],
                               c + '_pred': self.yhat_te[c]})
                 .sort_values(c + '_true')
                 )
            ax.scatter(residuals[f'{c}_true'],
                        residuals[f'{c}_pred'] - residuals[f'{c}_true'],
                        s=1, label=c, color=self.clrs['red'])
            ax.axhline(color='white'
                        if cfg.plot_cfg['style'].startswith('dark')
                        else 'black',
                        ls='--')
            ax.set_xlabel(r'$\vartheta_{}$'.format(self.param_map.get(c, 'x')) +
                       ' ground truth in °C')
            ax.set_title(r'$\vartheta_{0}$ prediction'.format(self.param_map.get(c, 'x')))
            if i == 0:
                ax.set_ylabel('prediction error\nin °C')
            ax.grid(alpha=0.5)

        fig.tight_layout()
        #fig.legend(markerscale=5.0, loc='lower left', ncol=len(self.actual))
        #plt.savefig(f'data/plots/residuals_{self.uid}.pdf', dpi=400)

    def print(self):
        print('')
        print('#' * 5 + ' Trial Report ' + '#'*5)
        print(f"Trial ID: {self.uid}")
        self.metrics = print_scores(self.actual, self.yhat_te)
        print('#' * 20)

