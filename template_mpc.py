import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
import do_mpc

def template_mpc(model, silence_solver=False):

    mpc = do_mpc.controller.MPC(model)
    # Set settings of MPC:
    mpc.settings.n_horizon =  20
    mpc.settings.n_robust =  0
    mpc.settings.open_loop =  0
    mpc.settings.t_step =  0.05
    mpc.settings.state_discretization =  'collocation'
    mpc.settings.collocation_type =  'radau'
    mpc.settings.collocation_deg =  3
    mpc.settings.collocation_ni =  1
    mpc.settings.store_full_solution =  True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    # setting up cost tracking 
    cost_tracking = model.aux['cost_tracking']

    # On récupère les vitesses articulaires pour le critère J2
    v3 = model.x['v3']
    v4 = model.x['v4']
    # definition du critere lterme critere de tracking, m term critere une fois arrivee a la cible 
    lterm = cost_tracking + (v3**2 + v4**2)
    mterm = cost_tracking  

    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Pénalité sur les entrées (rterm) pour lisser la commande
    mpc.set_rterm(
        u1=1,
        u2=1,
        u3=1,
        u4=1
    )
    # limite sur la commande 
    mpc.bounds['lower','_u', 'u1'] = -2.0
    mpc.bounds['upper','_u', 'u1'] =  2.0
    mpc.bounds['lower','_u', 'u2'] = -2.0
    mpc.bounds['upper','_u', 'u2'] =  2.0
    mpc.bounds['lower','_u', 'u3'] = -2.0
    mpc.bounds['upper','_u', 'u3'] =  2.0
    mpc.bounds['lower','_u', 'u4'] = -2.0
    mpc.bounds['upper','_u', 'u4'] =  2.0

    # limite sur les vitesses
    mpc.bounds['lower','_x', 'v1'] = -1.5
    mpc.bounds['upper','_x', 'v1'] =  1.5
    mpc.bounds['lower','_x', 'v2'] = -1.5
    mpc.bounds['upper','_x', 'v2'] =  1.5
    mpc.bounds['lower','_x', 'v3'] = -1.5
    mpc.bounds['upper','_x', 'v3'] =  1.5
    mpc.bounds['lower','_x', 'v4'] = -1.5
    mpc.bounds['upper','_x', 'v4'] =  1.5

    # limite sur les articulation 
    gamma = np.pi/2 # gamma a modifier 

    mpc.bounds['lower','_x', 'x4'] = -gamma
    mpc.bounds['upper','_x', 'x4'] =  gamma
    mpc.bounds['lower','_x', 'x5'] = -gamma
    mpc.bounds['upper','_x', 'x5'] =  gamma

    # avoide obstacles 
    
    mpc.set_nl_cons('safety_p0', -model.aux['cons_p0'], 0) # P0 > 0.4m 
    mpc.set_nl_cons('safety_p1', -model.aux['cons_p1'], 0) # P1 > 0.3m 
    mpc.set_nl_cons('safety_p2', -model.aux['cons_p2'], 0) # P2 > 0.1m 
    mpc.set_nl_cons('safety_pf', -model.aux['cons_pf'], 0) # Pf > 0.15m 

    # build du controlleur 

    mpc.setup()

    return mpc