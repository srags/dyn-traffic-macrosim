import numpy as np
import pandas as pd
import os

class METANET_Params:
    def __init__(self, path=None, control_h=None, num_timesteps=360, num_segments=14):
        if path is not None and control_h is not None:
            self.params = {
                "tau": None,
                "K": None,
                "eta_high": None,
                "p_crit": None,
                "v_free": None,
                "a": None,
                'q_capacity': None,
                'r': None,
                'beta': None,
                'gamma': None
            }
            
            num_params = int(num_timesteps/control_h)

            for i in range(1, num_params+1):
                fold_path = f'{path}/control_h_{control_h}/params_{i}'
                if os.path.exists(fold_path):
                    params_i = {
                        "tau": np.load(f'{fold_path}/tau.npy').reshape(-1),
                        "K": np.load(f'{fold_path}/K.npy').reshape(-1),
                        "eta_high": np.load(f'{fold_path}/eta_high.npy').reshape(-1),
                        "p_crit": np.load(f'{fold_path}/rho_crit.npy').reshape(-1),
                        "v_free": np.load(f'{fold_path}/v_free.npy').reshape(-1),
                        "a": np.load(f'{fold_path}/a.npy').reshape(-1),
                        'q_capacity': np.array([2200 for i in range(num_segments)])
                    }
                    try:
                        params_i['r'] = np.load(f'{fold_path}/r_inflow_array.npy')
                    except:
                        params_i['r'] = np.array([0 for i in range(num_segments)])
                    try:
                        params_i['beta'] = np.load(f'{fold_path}/beta_array.npy')
                    except:
                        params_i['beta'] = np.array([0 for i in range(num_segments)])
                    try:
                        params_i['gamma'] = np.load(f'{fold_path}/gamma_array.npy')
                    except:
                        params_i['gamma'] = np.array([1 for i in range(num_segments)])
   
                    for key in self.params.keys():
                        new_params = np.tile(params_i[key], (control_h, 1))
                        self.params[key] = np.vstack((self.params[key], new_params)) if self.params[key] is not None else new_params
                
            for key in self.params.keys():
                assert(self.params[key].shape[0] == num_timesteps)
            # Combine params from all foldes in path + "control_h_{control_h}"
            

        elif path is not None:
            self.params = {
                "tau": np.load(f'{path}/tau.npy'),
                "K": np.load(f'{path}/K.npy'),
                "eta_high": np.load(f'{path}/eta_high.npy'),
                "p_crit": np.load(f'{path}/rho_crit.npy'),
                "v_free": np.load(f'{path}/v_free.npy'),
                "a": np.load(f'{path}/a.npy'),
                'q_capacity': [2200 for i in range(num_segments)]
            }
            try:
                self.params['r'] = np.load(f'{path}/r_inflow_array.npy')
            except:
                self.params['r'] = np.array([0 for i in range(num_segments)])
            try:
                self.params['beta'] = np.load(f'{path}/beta_array.npy')
            except:
                self.params['beta'] = np.array([0 for i in range(num_segments)])
            try:
                self.params['gamma'] = np.load(f'{path}/gamma_array.npy')
            except:
                self.params['gamma'] = np.array([1 for i in range(num_segments)])
        else:
            # Use default
            self.params = {
                "tau": np.array([18/3600 for i in range(num_segments)]),
                "K": np.array([40 for i in range(num_segments)]),
                "eta_high": np.array([30 for i in range(num_segments)]),
                "p_crit": np.array([37.45 for i in range(num_segments)]),
                "v_free": np.array([120 for i in range(num_segments)]),
                "a": np.array([1.4 for i in range(num_segments)]),
                'q_capacity': np.array([2400 for i in range(num_segments)]),
                'r' : np.array([0 for i in range(num_segments)]),
                'beta' : np.array([0 for i in range(num_segments)]),
                'gamma' : np.array([1 for i in range(num_segments)])
            }

    def get_params(self):
        return self.params

    def get_param(self, key):
        return self.params.get(key, None)