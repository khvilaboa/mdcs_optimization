import random
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Set
from mercury import Mercury
from mercury.config import ResourceManagerConfig, TransceiverConfig
#from samples.models.pu_power.rx580_power_model import Rx580PowerModel
from functools import wraps
import os
import pickle

from samples.models.pu_power.rx580_power_model import Rx580PowerModel

plt.rcParams["font.family"] = "Times New Roman"

MIN_EPOCH = 0
SIM_TIME = 2000
LIMIT_UES = None
PUS_PER_EDC = 10
PATH_UES = "data/corr_coord/footstep_cl_corr.log"
PATH_APS = "data/corr_coord/aps_clus0.csv"
PATH_EDCS = "data/corr_coord/edcs_clus0_sel3.csv"

def ue_cache(prepare_ues_func):
    @wraps(prepare_ues_func)
    def with_cache(*args, **kwargs):
        os.makedirs("cache", exist_ok=True)
        n_ues = str(kwargs["limit_n_ues"]) if kwargs["limit_n_ues"] else "all"
        pickle_file = os.path.join("cache",
                                   os.path.splitext(os.path.basename(args[1]))[0] + "_" + str(n_ues) + ".pickle")

        if os.path.exists(pickle_file):
            print("Loading UEs from cache...")
            ue_mobilities = pickle.load(open(pickle_file, "rb"))
            print("UES loaded")
        else:
            #df = pd.read_csv(args[1], index_col=None)
            #df["cab_id"] = df["cab_id"].astype(str)
            # df["epoch"] = df["epoch"] - df["epoch"].min()
            print("Generating UEs...")
            ue_mobilities = prepare_ues_func(*args, **kwargs)
            print("UEs loaded (%d)." % len(ue_mobilities))
            pickle.dump(ue_mobilities, open(pickle_file, "wb"))
            print("UEs cache saved...")

        return ue_mobilities

    return with_cache

@ue_cache
def prepare_ues_config(services: Set[str], traces_path: str, t_start: float, limit_n_ues: int = None):
    ues_traces = pd.read_csv(traces_path, sep=',')

    # Generate lifetimes DataFrame
    ues_lifetime = ues_traces.groupby(["cab_id"]).agg({'epoch': [np.min, np.max]})
    ues_lifetime.columns = ues_lifetime.columns.droplevel()
    ues_lifetime = ues_lifetime.rename(columns={'amin': 't_start', 'amax': 't_end'})
    ues_lifetime.reset_index(level=0, inplace=True)  # index to column

    ue_configs = dict()
    for index, row in ues_lifetime.iterrows():
        if limit_n_ues is not None and index >= limit_n_ues:
            break

        history_df = ues_traces[ues_traces['cab_id'] == row['cab_id']][['epoch', 'x', 'y']]
        ue_configs[row['cab_id']] = {
            'ue_id': str(int(row['cab_id'])),
            'services_id': services,
            'radio_antenna': None,
            't_start': row['t_start'],
            't_end': row['t_end'],
            'mobility_name': 'history',
            'mobility_config': {
                't_start': t_start,
                't_column': 'epoch',
                'history': history_df
            }
        }


        print(index)
    return ue_configs


mercury = Mercury()
mercury.fog_model.set_max_guard_time(60)
mercury.fog_model.add_custom_pu_power_model('Rx580', Rx580PowerModel)

#######################
# Loading the models
#######################
# mercury.fog_model.add_custom_pu_power_model('Rx580', Rx580PowerModel)

# mercury.fog_model.define_p_unit_config(p_unit_id='pu_basic',  # The ID must be unique
#                                        max_u=100,  # Standard to Specific Utilization Factor
#                                        # dvfs_table={100: {'memory_clock': 2000, 'core_clock': 1366}},
#                                        t_on=6,  # Time required for switching on the processing unit
#                                        t_off=2,  # Time required for switching off the processing unit
#                                        #t_start=0.5,
#                                        #t_stop=0.2,
#                                        #t_operation=5,
#                                        pwr_model_name="idle_active",
#                                        pwr_model_config={"idle_power": 100, "active_power": 150})

#########################################
# Defining all the UE-related parameters
#########################################

# Services configuration
mercury.fog_model.define_service_config(service_id='pred_inference',  # Service ID must be unique
                                        header=18,  # header of any service package
                                        content=54,
                                        generation_time=lambda: 60,  # data stream size in bits per second
                                        min_closed_t=0,  # minimum time to stay closed for service sessions in seconds
                                        min_open_t=0.2,  # minimum time to stay open for service sessions in seconds
                                        service_timeout=0.3,  # service requests timeout in seconds
                                        window_size=1)  # Number of requests to be sent simultaneously with no acknowledgement

mercury.fog_model.define_service_config(service_id='pred_training',  # Service ID must be unique
                                        header=18,  # header of any service package
                                        content=0,
                                        generation_time=lambda: random.gauss(7200, 1800),  # data stream size in bits per second
                                        min_closed_t=0,  # minimum time to stay closed for service sessions in seconds
                                        min_open_t=60,  # minimum time to stay open for service sessions in seconds
                                        service_timeout=0.2,  # service requests timeout in seconds
                                        window_size=1)  # Number of requests to be sent simultaneously with no acknowledgement

# Creating and adding UEs
ue_configs = prepare_ues_config({'pred_inference', 'pred_training'}, PATH_UES, MIN_EPOCH, limit_n_ues=LIMIT_UES)

for ue_config in ue_configs.values():
    mercury.fog_model.add_ue_config(**ue_config)

####################################
# Defining standard configurations
####################################

# Define the Radio Access Network service parameters
mercury.fog_model.define_rac_config(header=0,
                                    # header size (in bits) of application messages related to radio access control
                                    pss_period=0,  # Primary Synchronization Signal message period (in s)
                                    rrc_period=0,  # Remote Resource Control message period (in s)
                                    timeout=0.2)  # Message timeout (in s) for application messages related to RAC

# Define network layer-level packets
mercury.fog_model.define_network_config(header=0)  # Network layer-level header size (in bits)

# Define edge federation management service parameters
mercury.fog_model.define_fed_mgmt_config(header=0,  # header size (in bits)
                                         content=0)  # EDC report size (in bits)

####################################
# Defining EDC-related parameters
####################################

edcs = pd.read_csv(PATH_EDCS).values

# Define Processing Units configurations
mercury.fog_model.define_pu_type_config(pu_type='Rx580',
                                        services={
                                            'pred_inference': {
                                                'utilization': 5,  # Computing resources per service session
                                                't_start': 0.05,  # Time required for starting a session
                                                't_stop': 0.05,  # Time required for stopping a session
                                                't_operation': 0.2,  # Time required for processing an operation
                                            },
                                            'pred_training': {
                                                'utilization': 20,  # Computing resources per service session
                                                't_start': 0.05,  # Time required for starting a session
                                                't_stop': 0.05,  # Time required for stopping a session
                                                't_operation': 0,  # Time required for processing an operation
                                            }
                                        },
                                        # dvfs_table={100: {'v': 1.5, 'f': 2000000000},
                                        #             90: {'v': 1.4, 'f': 1800000000},
                                        #             80: {'v': 1.3, 'f': 1600000000},
                                        #             70: {'v': 1.2, 'f': 1400000000},
                                        #             60: {'v': 1.1, 'f': 1200000000},
                                        #             50: {'v': 1.0, 'f': 1000000000},
                                        #             40: {'v': 0.9, 'f': 800000000}
                                        # },
                                        dvfs_table={100: {
                                            'memory_clock': 2000,
                                            'core_clock': 1366
                                        }},
                                        n_threads=0,  # Number of threads TODO
                                        t_on=10,  # Time required for switching on the PU
                                        t_off=2,  # Time required for switching off the PU
                                        pwr_model_name='Rx580',  # We want to use the custom Rx580 power model
                                        pwr_model_config={
                                            'file_path_to_model_and_scalers': '../models/pu_power/rx580_power_model/power_model_rx580',
                                        })

mercury.fog_model.define_edc_rack_cooler_type('immersion',
                                              pwr_model_name='2_phase_immersion',
                                              pwr_model_config={
                                                  'shaft_power': lambda x: 0,  # TODO
                                                  'efficiency': lambda x: 1,
                                                  'specific_heat': 4.184,
                                                  'density': 1,
                                                  't_difference': 20,
                                                  'min_flow_rate': 0,
                                              })

resource_manager_config = ResourceManagerConfig(disp_function_name='less_overall_pwr',  # Sessions to fullest PU
                                                hot_standby_name='session', # hot standby will be dynamic -> no need of defining it here
                                                hot_standby_config={
                                                    'min_sessions': {'pred_inference': 100, 'pred_training': 20},
                                                    'extra_sessions': {'pred_inference': 20, 'pred_training': 5}
                                                })

crosshaul_transceiver = TransceiverConfig(tx_power=10,              # transmitting power (in dBm)
                                              gain=0,                   # gain (in dB)
                                              noise_name='thermal',     # Noise model ID -> thermal
                                              default_eff=5)            # default spectral efficiency (in bps/Hz)

mercury.fog_model.add_core_config(core_location=(50, 50),  # ue_location of the core network elements (x, y) [m]
                  core_trx=crosshaul_transceiver,  # transceiver used by core network elements
                  sdn_strategy_name='closest')  # SDN strategy name

mercury.fog_model.add_edcs_controller_config()

# Define Crosshaul
mercury.fog_model.add_crosshaul_config()  # crosshaul physical packets header in bits

# Define Radio Layer
mercury.fog_model.add_radio_config()

for edc in edcs:
    mercury.fog_model.add_edc_config(edc_id=edc[0],  # the ID must be unique
                     edc_location=(edc[1], edc[2]),  # ue_location of the EDC (x, y) [m]
                     edc_trx=crosshaul_transceiver,  # Transceiver used by the EDC
                     r_manager_config=resource_manager_config,  # EDC Dispatcher configuration
                     edc_racks_map={'rack_1': ('immersion', {'pu_{}'.format(i): 'Rx580' for i in range(PUS_PER_EDC)})},
                     env_temp=298)  # EDC base temperature (not used yet)

#Defining APs
aps = pd.read_csv(PATH_APS).values

for ap in aps:
    mercury.fog_model.add_ap_config(ap_id=ap[0],  # Access Point ID must be unique
                    ap_location=(ap[1], ap[2]),  # Access Point location (x, y) [m]
                    )

# Add transducer for generating CSV results
mercury.add_transducers('csv', dir_path='results')

#mercury.fog_model_quick_build(lite=True, shortcut=False)
mercury.fog_model.build(lite=True, shortcut=False)
#mercury.fog_model.initialize_coordinator()
print(time.time())
mercury.fog_model.start_simulation(time_interv=SIM_TIME)

alpha = 1
stacked = True

mercury.plot_delay(alpha)
mercury.plot_edc_utilization(stacked, alpha)
mercury.plot_edc_power_demand(stacked, alpha)
mercury.plot_edc_it_power(stacked, alpha)
mercury.plot_edc_cooling_power(stacked, alpha)

print(time.time())
print('done')
