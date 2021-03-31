import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LSTM, GRU, RNN, \
    GaussianNoise, TimeDistributed, Input, add, Flatten, Dense, GRUCell, \
    Concatenate, BatchNormalization, Add, Dropout
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.optimizers as opts
from tensorflow.keras import regularizers, initializers, constraints, activations
import numpy as np
import warnings
from copy import deepcopy
import aux.config as cfg

warnings.filterwarnings('ignore')


def shifted_relu(x, shift=1e-3):
    return shift + tf.nn.relu(x)


def squished_sigmoid(x):
    return 2 * tf.nn.sigmoid(4 * x)


def double_swish(x):
    return x * x * tf.nn.sigmoid(x)


def sin_act(x):
    return tf.math.sin(x)


def cone(x):
    return tf.math.maximum(-x * 1e-3, x)


def squared(x):
    return x ** 2

def sigmoid_linear_overshoot(x):
    return tf.nn.sigmoid(x) + tf.nn.relu(x-2)

def biased_elu(x):
    return tf.nn.elu(x) + 1


sinus = layers.Activation(sin_act)
ssig = layers.Activation(squished_sigmoid)
srelu = layers.Activation(shifted_relu)
dswish = layers.Activation(double_swish)
_cone = layers.Activation(cone)
_squr = layers.Activation(squared)
_siglinover = layers.Activation(sigmoid_linear_overshoot)
_biasedelu = layers.Activation(biased_elu)
tf.keras.utils.get_custom_objects().update({'srelu': srelu,
                                            'sinus': sinus,
                                            'squished_sigmoid': ssig,
                                            'double_swish': dswish,
                                            'cone': _cone,
                                            'squared': _squr,
                                            'sig_lin_overshoot': _siglinover,
                                            'biased_elu': _biasedelu})


def build_lptn(x_shape, cell_cls,
               loss_weights=None,
               verbose=True, clipnorm=None, clipvalue=None,
               lr=1e-3, loss='mse', optimizer_cls=None, drop_g=None,
               x_cols=None, p_is_branchful=None, layer_cfg=None):
    p_is_branchful = p_is_branchful or False

    x = Input(batch_input_shape=x_shape)
    out = RNN(cell_cls(x_cols, p_is_branchful, layer_cfg, drop_g=drop_g),
              return_sequences=True, stateful=True, name='rnn')(x)

    model = Model(x, out)
    def weighted_mse(y_true, y_pred):
        """Weight first component 50%"""
        s = cfg.keras_cfg['tnn_params']['temp_scale']
        return tf.reduce_mean(#tf.constant([[0.5, 0.166, 0.166, 0.166]]) *
                             tf.square(s*(y_pred - y_true)))
    opts_d = {'adam': opts.Adam, 'nadam': opts.Nadam, 'adamax': opts.Adamax,
              'sgd': opts.SGD, 'rmsprop': opts.RMSprop}

    optimizer_cls = optimizer_cls or 'nadam'
    optimizer_cls = opts_d[optimizer_cls]
    optimizer = optimizer_cls(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue)
    if loss_weights is not None:
        loss_weights = list(loss_weights)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  sample_weight_mode='temporal',
                  loss_weights=loss_weights, run_eagerly=False)
    if verbose:
        model.summary()
    return model


def build_wallscheid_lptn(x_shape, loss_weights=None,
                          verbose=True,
                          loss='mse',
                          # LPTN params
                          cap0=np.log10(1.0666e4), cap1=np.log10(6.5093e3),
                          cap2=np.log10(0.437127e3), cap3=np.log10(3.5105e3),
                          const_Rs0=np.log10(0.0375),
                          const_Rs1=np.log10(0.0707),
                          const_Rs2=np.log10(0.0899),
                          lin_Rs_slope=-54e-4,
                          lin_Rs_bias=np.log10(18e-3),
                          exp_Rs_magn0=1.7275, exp_Rs_magn1=0.8486,
                          exp_Rs_magn2=0.6349,
                          exp_Rs_b0=0.1573, exp_Rs_b1=0.1428, exp_Rs_b2=0.1184,
                          exp_Rs_a0=0.3039, exp_Rs_a1=0.2319, exp_Rs_a2=0.1205,
                          bipoly_Rs_magn=np.log10(0.3528),
                          bipoly_Rs_a=-0.2484,
                          bipoly_Rs_b=0.0276,
                          bipoly_Rs_c=0.3331,
                          ploss_Rdc=np.log10(14.6e-3),
                          ploss_alpha_cu=20e-4,
                          ploss_alpha_ac_1=0.562,
                          ploss_alpha_ac_2=0.2407,
                          ploss_beta_cu=2.5667,
                          ploss_k_1_0=0.5441, ploss_k_1_1=78e-4,
                          ploss_k_1_2=0.0352, ploss_k_1_3=-0.7438,
                          ploss_k_2=0.8655,
                          ploss_alpha_fe=-28e-4, schlepp_factor=1.4762,
                          ):
    """O. Wallscheid, "Ein Beitrag zur thermischen Ausnutzung permanenterregter
    Synchronmotoren in automobilen Traktionsanwendungen", Dissertation 2017
    """
    x = Input(batch_input_shape=x_shape, ragged=False)
    out = RNN(WallscheidLPTNCell(cap0, cap1, cap2, cap3,
                                 const_Rs0, const_Rs1, const_Rs2,
                                 lin_Rs_slope,
                                 lin_Rs_bias,
                                 exp_Rs_magn0, exp_Rs_magn1, exp_Rs_magn2,
                                 exp_Rs_b0, exp_Rs_b1, exp_Rs_b2,
                                 exp_Rs_a0, exp_Rs_a1, exp_Rs_a2,
                                 bipoly_Rs_magn,
                                 bipoly_Rs_a,
                                 bipoly_Rs_b,
                                 bipoly_Rs_c,
                                 ploss_Rdc,
                                 ploss_alpha_cu,
                                 ploss_alpha_ac_1,
                                 ploss_alpha_ac_2,
                                 ploss_beta_cu,
                                 ploss_k_1_0, ploss_k_1_1,
                                 ploss_k_1_2, ploss_k_1_3,
                                 ploss_k_2,
                                 ploss_alpha_fe, schlepp_factor
                                 ),
              return_sequences=True, stateful=True, name='rnn')(x)

    if loss_weights is not None:
        loss_weights = list(loss_weights)

    model = Model(x, out)

    model.compile(loss=loss, sample_weight_mode='temporal',
                  loss_weights=loss_weights, run_eagerly=False)
    if verbose:
        model.summary()
    return model


class TNNCell(layers.Layer):
    """W. Kirchgässner, O. Wallscheid, J. Böcker, "Thermal Neural Networks:
    Lumped-Parameter Thermal Modeling with State-Space Machine Learning",
    arXiv:2103.16323 [cs.LG]
    """
    def __init__(self, x_cols, drop_g=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_time = cfg.data_cfg['downsample_rate'] * \
                           int(cfg.data_cfg['sample_time'][:-2]) / 1000
        self.state_size = None

        # indices
        self.x_cols = x_cols
        self.temp_idcs = [i for i, x in enumerate(x_cols) if x in
                          cfg.data_cfg['temperature_cols']]
        self.nontemp_idcs = [i for i, x in enumerate(x_cols) if x not in
                            cfg.data_cfg['temperature_cols']]

        self.output_size = len(cfg.data_cfg['Target_param_names'])

        # LPTN coefficients
        self.caps_net = self.add_weight(trainable=True,
                                   name='inv_caps', shape=(1, self.output_size),
                                   initializer=initializers.RandomNormal(-3, 0.5),
                                )

        self.drop_g = drop_g or []
        n_temps = len(cfg.data_cfg['temperature_cols'])
        n_conds = int(0.5 * n_temps * (n_temps - 1)) - len(self.drop_g)
        self.conductance_net = tf.keras.Sequential([
            Dense(units=n_conds, activation='sigmoid', name='g_1'),
        ])

        # populate adjacency matrix
        self.adj_mat = np.zeros((n_temps, n_temps), dtype=int)
        adj_idx_arr = np.ones_like(self.adj_mat)
        if len(self.drop_g) > 0:
            adj_idx_arr[tuple(zip(*self.drop_g))] = 0
        adj_idx_arr = adj_idx_arr[np.triu_indices(n_temps, 1)].ravel()
        self.adj_mat[np.triu_indices(n_temps, 1)] = np.cumsum(adj_idx_arr) - 1
        self.adj_mat += self.adj_mat.T
        self.n_temps = n_temps

        # power loss modeling
        self.ploss_out_gen = tf.keras.Sequential([
            Dense(16, name=f'ploss_gen_1', activation='tanh', use_bias=True),
            Dense(self.output_size, name=f'ploss_gen_2', use_bias=True,
                  activation='linear'),
            layers.Lambda(lambda x: tf.abs(x))
        ])

    def get_config(self):
        config = super().get_config().copy()
        config.update({'x_cols': self.x_cols,
                       'drop_g': self.drop_g})
        return config

    def build(self, inputs):
        self.state_size = self.output_size
        self.built = True
        self.trainable = True

    def call(self, inputs, states, training=True):
        prev_out = states[0]  # (49, 4)

        temps = tf.concat([prev_out] + [tf.reshape(inputs[:, i], [-1, 1])
                                        for i in self.temp_idcs], axis=1)
        non_temps = tf.stack([inputs[:, i] for i in self.nontemp_idcs], -1)
        all_inputs =  tf.concat([non_temps, temps], 1)
        # thermal conductances
        conducts = self.conductance_net(all_inputs)

        temp_diffs_l = [
            tf.reduce_sum(tf.stack([
                (temps[:, j] - prev_out[:, i]) *
                conducts[:, self.adj_mat[i, j]]
                for j in range(self.n_temps) if j != i
                                                and (i, j) not in self.drop_g
            ], -1), 1)
            for i in range(self.output_size)
        ]
        temp_diffs = tf.stack(temp_diffs_l, axis=-1)

        # powerloss
        power_loss = self.ploss_out_gen(all_inputs)

        out = prev_out + \
              self.sample_time * \
              (10**self.caps_net) * (temp_diffs + power_loss )

        return prev_out, tf.clip_by_value(out, -1, 5)


class ConfigurableTNNCell(TNNCell):
    def __init__(self, x_cols, p_is_branchful,
                 layer_cfg, drop_g=None, *args, **kwargs):
        super().__init__(x_cols, drop_g, *args, **kwargs)
        self.g_layer_cfg = layer_cfg['g']
        self.p_layer_cfg = layer_cfg['p']
        self.p_is_branchful = p_is_branchful

        # conductances
        self.conductance_net = tf.keras.Sequential([
            Dense(**l_d) for l_d in self.g_layer_cfg
        ])
        self.conductance_net.add(layers.Lambda(lambda x: tf.abs(x)))

    def build(self, inputs):
        super().build(inputs)
        # power loss
        if self.p_is_branchful:

            x = Input(batch_input_shape=(inputs[0],
                                         inputs[1] + self.output_size))
            out_l = []
            for t in cfg.data_cfg['Target_param_names']:
                interim_cfg = [deepcopy(p) for p in self.p_layer_cfg[:-1]]
                for l in interim_cfg:
                    t = t.replace('_[°C]', '')
                    l['name'] = f'{l["name"]}_{t}'
                out_l.append(tf.keras.Sequential([
                    Dense(**l_d) for l_d in interim_cfg
                ])(x))
            if len(out_l) > 1:
                y = Concatenate()(out_l)
            else:
                y = out_l[0]
            y = Dense(**self.p_layer_cfg[-1])(y)
            y = layers.Lambda(lambda z: tf.abs(z))(y)
            self.ploss_out_gen = Model(inputs=x, outputs=y)

        else:
            self.ploss_out_gen = tf.keras.Sequential([
                Dense(**l_d) for l_d in self.p_layer_cfg
            ])
            self.ploss_out_gen.add(layers.Lambda(lambda x: tf.abs(x)))

    def get_config(self):
        config = super().get_config()
        config.update({
            'p_is_branchful': self.p_is_branchful,
            'layer_cfg': {'g': self.g_layer_cfg, 'p': self.p_layer_cfg},
        })
        return config


class WallscheidLPTNCell(Layer):

    def __init__(self,
                 # lptn params
                 cap0, cap1, cap2, cap3,
                 const_Rs0, const_Rs1, const_Rs2,
                 lin_Rs_slope,
                 lin_Rs_bias,
                 exp_Rs_magn0, exp_Rs_magn1, exp_Rs_magn2,
                 exp_Rs_b0, exp_Rs_b1, exp_Rs_b2,
                 exp_Rs_a0, exp_Rs_a1, exp_Rs_a2,
                 bipoly_Rs_magn,
                 bipoly_Rs_a,
                 bipoly_Rs_b,
                 bipoly_Rs_c,
                 ploss_Rdc,
                 ploss_alpha_cu,
                 ploss_alpha_ac_1,
                 ploss_alpha_ac_2,
                 ploss_beta_cu,
                 ploss_k_1_0, ploss_k_1_1,
                 ploss_k_1_2, ploss_k_1_3,
                 ploss_k_2,
                 ploss_alpha_fe,
                 schlepp_factor,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.sample_time = 0.5
        self.state_size = None

        # indices
        x_cols = cfg.data_cfg['Input_param_names']
        self.ambient_idx = x_cols.index('ambient')
        self.coolant_idx = x_cols.index('coolant')
        self.motor_speed_idx = x_cols.index('motor_speed')
        self.i_d_idx = x_cols.index('i_d')
        self.i_q_idx = x_cols.index('i_q')
        self.i_idx = x_cols.index('i_s')
        self.ploss_fe_idx = x_cols.index('iron_loss')
        self.schlepp_idx = x_cols.index('schlepp')

        y_cols = cfg.data_cfg['Target_param_names']
        self.pm_idx = y_cols.index('pm')
        self.sy_idx = y_cols.index('stator_yoke')
        self.st_idx = y_cols.index('stator_tooth')
        self.sw_idx = y_cols.index('stator_winding')
        self.output_size = len(y_cols)

        # LPTN params
        #  O. Wallscheid, "Ein Beitrag zur thermischen Ausnutzung permanenterregter
        #  Synchronmotoren in automobilen Traktionsanwendungen"
        # log transform selected parameters
        cap0 = 10 ** cap0
        cap1 = 10 ** cap1
        cap2 = 10 ** cap2
        cap3 = 10 ** cap3
        const_Rs0 = 10 ** const_Rs0
        const_Rs1 = 10 ** const_Rs1
        const_Rs2 = 10 ** const_Rs2
        lin_Rs_bias = 10 ** lin_Rs_bias
        bipoly_Rs_magn = 10 ** bipoly_Rs_magn
        ploss_Rdc = 10 ** ploss_Rdc

        self.capacities = tf.constant([cap0, cap1, cap2, cap3],
                                      dtype=tf.float32)
        self.const_Rs = tf.constant([const_Rs0, const_Rs1, const_Rs2],
                                    dtype=tf.float32)
        self.lin_Rs_slope = tf.constant(lin_Rs_slope,
                                        dtype=tf.float32)  # alpha_K,SR
        self.lin_Rs_bias = tf.constant(lin_Rs_bias, dtype=tf.float32)
        self.exp_Rs_magn = tf.constant(
            [exp_Rs_magn0, exp_Rs_magn1, exp_Rs_magn2], dtype=tf.float32)
        self.exp_Rs_b = tf.reshape(
            tf.constant([exp_Rs_b0, exp_Rs_b1, exp_Rs_b2], dtype=tf.float32),
            [1, 3])
        self.exp_Rs_a = tf.constant([exp_Rs_a0, exp_Rs_a1, exp_Rs_a2],
                                    dtype=tf.float32)
        self.bipoly_Rs_magn = tf.constant(bipoly_Rs_magn, dtype=tf.float32)
        self.bipoly_Rs_a = tf.constant(bipoly_Rs_a, dtype=tf.float32)
        self.bipoly_Rs_b = tf.constant(bipoly_Rs_b, dtype=tf.float32)
        self.bipoly_Rs_c = tf.constant(bipoly_Rs_c, dtype=tf.float32)
        self.ploss_Rdc = tf.constant(ploss_Rdc, dtype=tf.float32)
        self.ploss_alpha_cu = tf.constant(ploss_alpha_cu, dtype=tf.float32)
        self.ploss_alpha_ac_1 = tf.constant(ploss_alpha_ac_1, dtype=tf.float32)
        self.ploss_alpha_ac_2 = tf.constant(ploss_alpha_ac_2, dtype=tf.float32)
        self.ploss_beta_cu = tf.constant(ploss_beta_cu, dtype=tf.float32)
        self.ploss_k_1 = tf.constant([ploss_k_1_0, ploss_k_1_1, ploss_k_1_2,
                                      ploss_k_1_3], dtype=tf.float32)
        self.ploss_k_2 = tf.constant(ploss_k_2, dtype=tf.float32)
        self.ploss_alpha_fe = tf.constant(ploss_alpha_fe, dtype=tf.float32)
        self.schlepp_factor = tf.constant(schlepp_factor, dtype=tf.float32)

        self.track = {}

    def build(self, input_shape):
        self.state_size = self.output_size
        self.built = True
        self.trainable = False

    def call(self, inputs, states):
        prev_out = states[0]  # (49, 4)

        ambient = inputs[:, self.ambient_idx]
        coolant = inputs[:, self.coolant_idx]
        motor_speed = tf.abs(inputs[:, self.motor_speed_idx]) / 6000  # normed
        current = inputs[:, self.i_idx]  # RMS

        # constant resistances
        r_sy_sw = self.const_Rs[0]
        r_sy_st = self.const_Rs[1]
        r_sw_st = self.const_Rs[2]

        # linear resistances
        r_c_sy = self.lin_Rs_bias * (1 + self.lin_Rs_slope * (coolant - 20))

        # exponentially decaying resistance
        reshaped_motor_speed = tf.repeat(tf.reshape(motor_speed, [-1, 1]),
                                         repeats=3, axis=1)
        r_exp = self.exp_Rs_magn * tf.exp(
            -reshaped_motor_speed / self.exp_Rs_b) + self.exp_Rs_a
        r_st_pm = r_exp[:, 0]
        r_sw_pm = r_exp[:, 1]
        r_pm_amb = r_exp[:, 2]

        # bivariate polynomial 1st order resistance
        normed_coolant = coolant / 100
        r_pm_c = self.bipoly_Rs_magn + \
                 self.bipoly_Rs_a * motor_speed + \
                 self.bipoly_Rs_b * normed_coolant + \
                 self.bipoly_Rs_c * motor_speed * normed_coolant
        r_pm_c = tf.math.maximum(tf.reshape(r_pm_c, [-1]), 1e-6)

        prev_pm = prev_out[:, self.pm_idx]
        prev_sy = prev_out[:, self.sy_idx]
        prev_st = prev_out[:, self.st_idx]
        prev_sw = prev_out[:, self.sw_idx]

        ploss_Rac = self.ploss_Rdc * (1 + self.ploss_alpha_ac_1 * motor_speed +
                                      self.ploss_alpha_ac_2 * tf.square(
                    motor_speed))
        lin_ploss_sw = (1 + self.ploss_alpha_cu * (prev_sw - 70))
        ploss_dc_ref = 3 * self.ploss_Rdc * tf.square(current)
        r_ac_over_dc_m1 = (ploss_Rac / self.ploss_Rdc - 1)
        ploss_cu_sw_ref = ploss_dc_ref * (1 + r_ac_over_dc_m1)

        ploss_sw = ploss_dc_ref * lin_ploss_sw + ploss_cu_sw_ref * \
                   (r_ac_over_dc_m1 / tf.maximum(
                       tf.math.maximum(tf.abs(lin_ploss_sw),
                                       1e-4) ** self.ploss_beta_cu,
                       1e-5))

        # LUT consists of iron loss, copper loss and mechanical loss
        ploss_fe = inputs[:, self.ploss_fe_idx] - \
                   self.schlepp_factor * inputs[:, self.schlepp_idx] - \
                   ploss_cu_sw_ref
        normed_current = current / 256
        k1 = tf.clip_by_value(self.ploss_k_1[0] +
                              self.ploss_k_1[1] * motor_speed +
                              self.ploss_k_1[2] * normed_current +
                              self.ploss_k_1[3] * motor_speed * normed_current,
                              0, 1)
        ploss_pm = (1 - k1) * ploss_fe * (
                1 + self.ploss_alpha_fe * (prev_pm - 63))
        ploss_sy = self.ploss_k_2 * k1 * ploss_fe * (
                1 + self.ploss_alpha_fe * (prev_sy - 55))
        ploss_st = (1 - self.ploss_k_2) * k1 * ploss_fe * (
                1 + self.ploss_alpha_fe * (prev_st - 65))

        ploss_temps = tf.stack([ploss_pm, ploss_sy, ploss_st, ploss_sw],
                               axis=-1)

        pm_diffs = (prev_st - prev_pm) / r_st_pm + \
                   (prev_sw - prev_pm) / r_sw_pm + \
                   (ambient - prev_pm) / r_pm_amb + \
                   (coolant - prev_pm) / r_pm_c
        sy_diffs = (prev_sw - prev_sy) / r_sy_sw + \
                   (prev_st - prev_sy) / r_sy_st + \
                   (coolant - prev_sy) / r_c_sy
        st_diffs = (prev_sy - prev_st) / r_sy_st + \
                   (prev_sw - prev_st) / r_sw_st + \
                   (prev_pm - prev_st) / r_st_pm
        sw_diffs = (prev_sy - prev_sw) / r_sy_sw + \
                   (prev_st - prev_sw) / r_sw_st + \
                   (prev_pm - prev_sw) / r_sw_pm
        temp_diffs = tf.stack([pm_diffs, sy_diffs, st_diffs, sw_diffs],
                              axis=-1)
        # DEBUG
        if hasattr(ploss_temps, 'numpy'):
            if len(self.track) == 0:
                for k in ['r_c_sy', 'r_pm_amb', 'r_pm_c', 'r_st_pm',
                          'r_sw_pm', 'r_sw_st', 'r_sy_st', 'r_sy_sw',
                          'pm_diffs', 'sy_diffs', 'st_diffs', 'sw_diffs',
                          'ploss_pm', 'ploss_sy', 'ploss_st', 'ploss_sw']:
                    self.track[k] = []
            self.track['r_c_sy'].append(r_c_sy.numpy())
            self.track['r_pm_amb'].append(r_pm_amb.numpy())
            self.track['r_pm_c'].append(r_pm_c.numpy())
            self.track['r_st_pm'].append(r_st_pm.numpy())
            self.track['r_sw_pm'].append(r_sw_pm.numpy())
            self.track['r_sw_st'].append(r_sw_st.numpy())
            self.track['r_sy_st'].append(r_sy_st.numpy())
            self.track['r_sy_sw'].append(r_sy_sw.numpy())

            self.track['pm_diffs'].append(pm_diffs.numpy())
            self.track['sy_diffs'].append(sy_diffs.numpy())
            self.track['st_diffs'].append(st_diffs.numpy())
            self.track['sw_diffs'].append(sw_diffs.numpy())

            self.track['ploss_pm'].append(ploss_pm.numpy())
            self.track['ploss_sy'].append(ploss_sy.numpy())
            self.track['ploss_st'].append(ploss_st.numpy())
            self.track['ploss_sw'].append(ploss_sw.numpy())

        ode_rh = (temp_diffs + ploss_temps) / self.capacities
        # euler forward
        out = prev_out + self.sample_time * ode_rh  # integrate

        # y_{k+1} = y_k + T_s * x
        return prev_out, tf.clip_by_value(out, 0, 200)
