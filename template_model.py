import numpy as np
import sys
from casadi import *
import do_mpc
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
def template_model():
    # define model type 
    model_type = "continuous"
    model = do_mpc.model.Model(model_type)
    # Define model parameters 
    l1 = 0.4
    l2 = 0.4
    l3 = 0.3
    r_obs = 0.4
    x_obs = 0.0
    y_obs = 0.0
    r_ref = 0.6
    alpha = np.pi
    theta_ref = alpha + np.pi # orientation vers l'obstacle
    x_ref = r_ref*cos(alpha)
    y_ref = r_ref*sin(alpha)

    # reference (setpoint)

    # state vector 
    x1 = model.set_variable('_x', 'x1')
    x2 = model.set_variable('_x', 'x2')
    x3 = model.set_variable('_x', 'x3')
    x4 = model.set_variable('_x', 'x4')
    x5 = model.set_variable('_x', 'x5')
    v1 = model.set_variable('_x', 'v1')
    v2 = model.set_variable('_x', 'v2')
    v3 = model.set_variable('_x', 'v3')
    v4 = model.set_variable('_x', 'v4')

    # algebric states 
    dx1 = model.set_variable('_z', 'dx1')
    dx2 = model.set_variable('_z','dx2')

    # inputs 
    u1 = model.set_variable('_u', 'u1')
    u2 = model.set_variable('_u', 'u2')
    u3 = model.set_variable('_u', 'u3')
    u4 = model.set_variable('_u', 'u4')

    # differential equations 
    model.set_rhs('x1',dx1)
    model.set_rhs('x2',dx2)
    model.set_rhs('x3',v2)
    model.set_rhs('x4',v3)
    model.set_rhs('x5',v4)
    model.set_rhs('v1',u1)
    model.set_rhs('v2',u2)
    model.set_rhs('v3',u3)
    model.set_rhs('v4',u4)

    # equation d'etat f(x,u,z) = 0 
    eq_etat = vertcat(
        # 1
        dx1 - v1*cos(x3),
        # 2
        dx2 - v1*sin(x3),

    )

    model.set_alg('eq_etat',eq_etat)

    # calcul des noeuds 
    p0_x = x1
    p0_y = x2

    # P1 
    p1_x = p0_x + l1 * cos(x3)
    p1_y = p0_y + l1 * sin(x3)

    # P2 
    p2_x = p1_x + l2 * cos(x3+x4)
    p2_y = p1_y + l2 * sin(x3+x4)

    # Pf 
    pf_x = p2_x + l3 * cos(x3+x4+x5)
    pf_y = p2_y + l3 * sin(x3+x4+x5)

    # Orientation finale de la pince (theta)
    theta_f = x3+x4+x5

    # Sauvegarde des expressions pour pouvoir les plotter plus tard
    model.set_expression('pf_x', pf_x)
    model.set_expression('pf_y', pf_y)

    # distance par rapport a l'obstacle superieur 

    d0 = sqrt((p0_x - x_obs)**2 + (p0_y - y_obs)**2) - r_obs
    d1 = sqrt((p1_x - x_obs)**2 + (p1_y - y_obs)**2) - r_obs
    d2 = sqrt((p2_x - x_obs)**2 + (p2_y - y_obs)**2) - r_obs
    df = sqrt((pf_x - x_obs)**2 + (pf_y - y_obs)**2) - r_obs

    # Definition des contraintes
    
    # P0 > 0.4m 
    cons_p0 = d0 - 0.4
    
    # P1 > 0.3m [cite: 588]
    cons_p1 = d1- 0.3
    
    # P2 > 0.1m [cite: 589]
    cons_p2 = d2 - 0.1
    
    # Pf > 0.15m [cite: 590]
    cons_pf = df - 0.15

    # On enregistre ces expressions pour le contrôleur
    model.set_expression('cons_p0', cons_p0)
    model.set_expression('cons_p1', cons_p1)
    model.set_expression('cons_p2', cons_p2)
    model.set_expression('cons_pf', cons_pf)

    # Definition du critere 

    cost_pos = (pf_x - x_ref)**2 + (pf_y - y_ref)**2
    
    # Erreur d'orientation (theta - theta_ref)^2
    cost_theta = (theta_f - theta_ref)**2

    # Terme total 
    cost_tracking = cost_pos + cost_theta

    model.set_expression('cost_tracking', cost_tracking)

    model.setup()

    return model

    

