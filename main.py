import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
import do_mpc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import time
def main():
    # Plot settings
    rcParams['text.usetex'] = False
    rcParams['axes.grid'] = True
    rcParams['lines.linewidth'] = 2.0
    rcParams['axes.labelsize'] = 'xx-large'
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'

    # local imports
    from template_mpc import template_mpc
    from template_model import template_model
    from template_simulator import template_simulator

    # user settings
    show_animation = True
    store_animation = False
    store_results = False

    # Setting up model
    model = template_model(alpha_ref=np.pi,r_ref=0.6,gamma=2*np.pi)

    # setting up a simulator, given the model
    simulator = template_simulator(model)

    # setting up controlleur
    mpc = template_mpc(model,n_horizon=40,step_time=0.05,criter=1) # cirtere = 0 on choisit le premier critere J1 = ||pf - pref||^2 + ||thetaf - theta_ref||^2, critere = 1 on choisit J2

    # setting up estimator 
    estimator = do_mpc.estimator.StateFeedback(model) #on suppose un retour d'etat (tout les etats sont mesurables)

    # definition des poistions initialles

    simulator.x0['x1'] = -2
    simulator.x0['x2'] = -2
    simulator.x0['x3'] = np.pi/2
    simulator.x0['x4'] = 0
    simulator.x0['x5'] = 0

    # extracting initial position from the simulator
    x0 = simulator.x0.cat.full()

    mpc.x0 = x0
    estimator.x0 = x0

    # setting up initial guesses
    mpc.set_initial_guess()

    # gets the initial condition of the algebraic variables
    z0 = simulator.init_algebraic_variables()

    n_steps = 200 
        
    print("Début de la simulation...")
    
    # plt.ion() 

    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        
        y_next = simulator.make_step(u0)
        
        x0 = estimator.make_step(y_next)
        
        if k % 10 == 0:
            print(f"Pas {k}/{n_steps} effectué.")

    print("Simulation terminée !")

    # tracee des resultat 

    data = mpc.data
    t = data['_time']
    
    x_obs, y_obs = 0.0, 0.0
    r_obs = 0.4
    r_ref = 0.6
    alpha_ref = np.pi
    
    x_target = r_ref * np.cos(alpha_ref)
    y_target = r_ref * np.sin(alpha_ref)
    theta_target = alpha_ref + np.pi

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Résultats de Simulation : Manipulateur Mobile MPC", fontsize=16)

    # cost function
    cost_val = data['_aux', 'cost_tracking']
    ax[0,0].plot(t, cost_val, 'k-', linewidth=2)
    ax[0,0].set_title("Critère de Coût (Tracking)", fontsize=12)
    ax[0,0].set_xlabel("Temps [s]")
    ax[0,0].set_ylabel("Erreur quadratique (Position + Orientation)")
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')

    # trajectoire
    pf_x = data['_aux', 'pf_x']
    pf_y = data['_aux', 'pf_y']
    
    # obstacle 
    circle_obs = plt.Circle((x_obs, y_obs), r_obs, color='r', alpha=0.3, label='Obstacle')
    ax[0,1].add_patch(circle_obs)
    
    # Reference
    circle_ref = plt.Circle((x_obs, y_obs), r_ref, color='b', fill=False, linestyle='--', label='Cercle Référence')
    ax[0,1].add_patch(circle_ref)
    
    # Trajectoire de la pince
    ax[0,1].plot(pf_x, pf_y, 'g-', linewidth=2, label='Trajectoire Pince')
    ax[0,1].plot(pf_x[0], pf_y[0], 'go', label='Départ') # Point de départ
    ax[0,1].plot(x_target, y_target, 'rx', markersize=12, markeredgewidth=3, label='Cible (Ref)') # Cible
    
    ax[0,1].set_title("Trajectoire XY de l'Effecteur", fontsize=12)
    ax[0,1].set_xlabel("X [m]")
    ax[0,1].set_ylabel("Y [m]")
    ax[0,1].axis('equal') # Important pour ne pas déformer les cercles
    ax[0,1].legend(loc='upper right')
    ax[0,1].grid(True)

    # orientation pince 
    x3 = data['_x', 'x3']
    x4 = data['_x', 'x4']
    x5 = data['_x', 'x5']
    theta_f_actual = x3 + x4 + x5
    
    ax[1,0].plot(t, theta_f_actual, 'b-', linewidth=2, label='Orientation Réelle')
    ax[1,0].axhline(theta_target, color='r', linestyle='--', linewidth=2, label='Référence')
    
    ax[1,0].set_title("Orientation de la Pince (Theta)", fontsize=12)
    ax[1,0].set_xlabel("Temps [s]")
    ax[1,0].set_ylabel("Angle [rad]")
    ax[1,0].legend()
    ax[1,0].grid(True)

    # distance de securite (distance obstacle)
    d0 = data['_aux', 'cons_p0']
    d1 = data['_aux', 'cons_p1']
    d2 = data['_aux', 'cons_p2']
    df = data['_aux', 'cons_pf']
    
    ax[1,1].plot(t, d0, label='Marge P0 (>0.4m)')
    ax[1,1].plot(t, d1, label='Marge P1 (>0.3m)')
    ax[1,1].plot(t, d2, label='Marge P2 (>0.1m)')
    ax[1,1].plot(t, df, label='Marge Pf (>0.15m)')
    
    # Ligne rouge à 0
    ax[1,1].axhline(0, color='r', linewidth=2, linestyle='--', label='LIMITE')
    
    ax[1,1].set_title("Marges de Sécurité (Doivent rester > 0)", fontsize=12)
    ax[1,1].set_xlabel("Temps [s]")
    ax[1,1].set_ylabel("Distance relative [m]")
    ax[1,1].legend(loc='best', fontsize='small')
    ax[1,1].grid(True)

    plt.tight_layout()
    plt.show(block=True)
    from matplotlib.animation import FuncAnimation, PillowWriter

    # 1. Fonction géométrique (Similaire à pendulum_bars dans l'exemple DIP)
    def get_robot_geometry(state):
        """
        Calcule la géométrie du robot.
        Dessine le châssis comme un RECTANGLE orienté.
        """
        # Paramètres géométriques
        l1, l2, l3 = 0.4, 0.4, 0.3
        width_chassis = 0.4  # Largeur visuelle du chariot
        
        # États
        x1, x2, x3 = state[0], state[1], state[2] # P0 (x,y) et orientation
        x4, x5 = state[3], state[4]               # Angles bras
        
        # --- 1. Calcul du Châssis (Rectangle) ---
        # Vecteur unitaire vers l'avant (direction du robot)
        vec_forward = np.array([np.cos(x3), np.sin(x3)])
        # Vecteur unitaire vers la droite (perpendiculaire)
        vec_right = np.array([np.sin(x3), -np.cos(x3)])
        
        # Centre arrière (P0)
        p0 = np.array([x1, x2])
        # Centre avant (P1)
        p1 = p0 + l1 * vec_forward
        
        # On définit les 4 coins du rectangle autour de l'axe P0-P1
        # On ajoute une petite marge à l'arrière (-0.1) et à l'avant (+0.1) pour que ce soit joli
        rear_offset = -0.1 * vec_forward
        front_offset = 0.1 * vec_forward
        
        # Coin Arrière-Droit
        c1 = p0 + rear_offset + (width_chassis/2) * vec_right
        # Coin Avant-Droit
        c2 = p1 + front_offset + (width_chassis/2) * vec_right
        # Coin Avant-Gauche
        c3 = p1 + front_offset - (width_chassis/2) * vec_right
        # Coin Arrière-Gauche
        c4 = p0 + rear_offset - (width_chassis/2) * vec_right
        
        # On ferme la boucle (C1 -> C2 -> C3 -> C4 -> C1)
        line_chassis_x = [c1[0], c2[0], c3[0], c4[0], c1[0]]
        line_chassis_y = [c1[1], c2[1], c3[1], c4[1], c1[1]]

        # --- 2. Calcul des Bras ---
        # P2 (Coude)
        p2 = p1 + l2 * np.array([np.cos(x3 + x4), np.sin(x3 + x4)])
        
        # Pf (Effecteur)
        pf = p2 + l3 * np.array([np.cos(x3 + x4 + x5), np.sin(x3 + x4 + x5)])
        
        # Ligne Bras 1 (P1 -> P2)
        line_arm1_x = [p1[0], p2[0]]
        line_arm1_y = [p1[1], p2[1]]
        
        # Ligne Bras 2 (P2 -> Pf)
        line_arm2_x = [p2[0], pf[0]]
        line_arm2_y = [p2[1], pf[1]]
        
        return (line_chassis_x, line_chassis_y), (line_arm1_x, line_arm1_y), (line_arm2_x, line_arm2_y)

    # 2. Préparation de la figure
    fig_anim, ax_anim = plt.subplots(figsize=(10, 10))

    # Dessin des éléments statiques
    circle_obs = plt.Circle((x_obs, y_obs), r_obs, color='r', alpha=0.3)
    circle_ref = plt.Circle((x_obs, y_obs), r_ref, color='b', fill=False, linestyle='--')
    ax_anim.add_patch(circle_obs)
    ax_anim.add_patch(circle_ref)
    ax_anim.plot(x_target, y_target, 'rx', markersize=10, markeredgewidth=3, label='Cible')

    # Initialisation des lignes du robot (vides au départ)
    line_chassis, = ax_anim.plot([], [], 'k-', linewidth=4, label='Châssis') # Noir épais
    line_arm1, = ax_anim.plot([], [], 'b-', linewidth=3, label='Bras 1')     # Bleu
    line_arm2, = ax_anim.plot([], [], 'g-', linewidth=3, label='Bras 2')     # Vert
    
    # Marqueurs pour les articulations
    joints, = ax_anim.plot([], [], 'ko', markersize=8) 

    # Réglage des axes (Fixes pour bien voir le mouvement)
    ax_anim.set_xlim(-2.5, 2.5)
    ax_anim.set_ylim(-2.5, 2.5)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.legend()
    ax_anim.set_title(f"Simulation Manipulateur Mobile (t=0.00s)")

    # 3. Fonction d'animation (appelée à chaque frame)
    x_arr = mpc.data['_x'] # Toutes les données de positions
    n_frames = x_arr.shape[0]

    def update(t_ind):
        # Récupère la géométrie pour l'instant t_ind
        state = x_arr[t_ind]
        (cx, cy), (a1x, a1y), (a2x, a2y) = get_robot_geometry(state)
        
        # Met à jour les lignes
        line_chassis.set_data(cx, cy)
        line_arm1.set_data(a1x, a1y)
        line_arm2.set_data(a2x, a2y)
        
        # Met à jour les points (articulations P0, P1, P2, Pf)
        # On concatène toutes les coords x et y
        all_x = [cx[0], cx[1], a1x[1], a2x[1]] 
        all_y = [cy[0], cy[1], a1y[1], a2y[1]]
        joints.set_data(all_x, all_y)
        
        # Met à jour le titre avec le temps
        current_time = t_ind * 0.05 # t_step
        ax_anim.set_title(f"Simulation Manipulateur Mobile (t={current_time:.2f}s)")
        
        return line_chassis, line_arm1, line_arm2, joints

    # 4. Création et Sauvegarde
    anim = FuncAnimation(fig_anim, update, frames=n_frames, interval=50, blit=False)
    
    # Sauvegarde en GIF avec Pillow (pas besoin d'installer ImageMagick)
    writer = PillowWriter(fps=20) 
    anim.save('simulation_robot.gif', writer=writer)
    
    print("GIF sauvegardé sous 'simulation_robot.gif' !")
    plt.show(block=True)

if __name__ == "__main__":
    main()