import numpy as np
from casadi import *
import do_mpc

def template_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'idas',
        'abstol': 1e-8,
        'reltol': 1e-8,
        't_step': 0.05
    }
    
    simulator.set_param(**params_simulator)

    # setup simulator 
    simulator.setup()

    return simulator