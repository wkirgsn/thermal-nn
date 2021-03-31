"""Configuration file"""

debug_cfg = {'DEBUG': False}

data_cfg = {
    'Input_param_names': ['ambient',
                          'coolant',
                          'u_d',
                          'u_q',
                          'motor_speed',
                          #'torque',
                          'i_d',
                          'i_q'
                           ],
    'Target_param_names': ['pm',
                           'stator_yoke',
                           'stator_tooth',
                           'stator_winding'
                          ],
    'temperature_cols': ['pm', 'stator_yoke',
                         'stator_tooth',
                         'stator_winding',
                         'ambient', 'coolant'],
    # for ewm and statistical moments
    'rolling_lookback':  [6360, 3360, 1320, 9480],
    'valset': ['67', '71', '78', '4'],  #, ['58']
    'testset': ['60', '62', '74'],  # , ['65', '72']
    'loss': 'mse',
    # profile ids to drop (don't need to actually exist)
    'drop': None,#['31', '36', '47', '46'],
    'subsequences': False,
    'subsequence_length': 3,  # in hours
    # paths
    'file_path': "data/input/measures_v2.csv",
    'random_seed': 2019,
    'downsample_rate': 1,
    'sample_time': '500ms',
    'downsample_val_test': True,
    'p_id_col': 'profile_id'
    }

plot_cfg = {'do_plot': True,
            'style': 'seaborn-whitegrid',
            }

keras_cfg = {
    'early_stop_patience': 30,
    'n_trials': 10,
    # the following flag will override params-batch_size with num profiles
    'full_data_as_batch': True,  # shorter profiles will be padded with zero
    'tbptt_len': 1227,  # in half-seconds
    'rnn_params': {'epochs': 120,
                   'activity_reg': 0.001,
                   'arch': 'res_lstm',
                   'bias_reg': 0.001,
                   'clipnorm': 1,
                   'clipvalue': 2,
                   'dropout_rate': 0.0,
                   'gauss_noise_std': 0.000001,
                   'kernel_reg': 0.001,
                   'lr_rate': 11e-3,
                   'n_layers': 2,
                   'n_units': 8,
                   'optimizer': 'adam',
                   'recurrent_reg': 0.01,
                   'multi_headed': False,
                   'reduce_weight_on_thinner_batches': False},
    'tnn_params': {'temp_scale': 100,
                   'layer_cfg': {
                            'p': [dict(units=16, name='p_1', activation='sigmoid'),
                                  dict(units=len(data_cfg['Target_param_names']),
                                       name='p_2', activation='sigmoid')],
                            'g':  [dict(units=2, name='g_1', activation='sinus'),
                                   dict(units=15, name='g_2', activation='biased_elu')]
                   }

    },
}
