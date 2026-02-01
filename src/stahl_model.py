import numpy as np
import itertools
import matplotlib.pyplot as plt


M = np.array([
                [1, 0, 0, 0, 0], 
                [2, 2, 0, 0, 0], 
                [3, 2, 3, 1, 0],
                [4, 4, 4, 3, 1], # S3 (Le piège attractif)
                [0, 1, 2, 4, 6]  # le 4 dans la 4ememcolone permet de rendre cette strat meilleure que celle juste au essus 
            ])

cardE = M.shape[0]
PREFERENCES_TABLE = np.array( list(itertools.permutations(np.arange(cardE))) )
                                                # Exemple d'accès
                                                # k = 0 -> (0, 1, 2, 3, 4)
                                                # k = 119 -> (4, 3, 2, 1, 0)
Nb_preferences2 = len(PREFERENCES_TABLE)
N = 4 # Nombre de niveaux d'intelligence

def calculer_sigma_n(p_inf_n, M, preferences_table):
    """
    Détermine les choix individuels (sigma) pour un niveau d'intelligence n donné.
    
    1. Calcule l'espérance de gain contre les joueurs moins intelligents (p_inf_n).
    2. Identifie les stratégies rationnelles (Rn).
    3. Sélectionne pour chaque type k la meilleure stratégie disponible dans Rn.

    Args:
        p_inf_n (np.array): Vecteur de probabilité (taille 5) représentant le jeu moyen des niveaux < n.
        M (np.array): Matrice des gains (5x5).
        preferences_table (np.array): Matrice (120x5) des ordres de préférence.

    Returns:
        sigma_n (np.array): Vecteur (taille 120) contenant l'action choisie par chaque type k.
        strategies_rationnelles (list): Liste des actions valides pour ce tour intitulee Rn dans l'article.
    """
    esperance_gains = M @ p_inf_n
    
    gain_max = np.max(esperance_gains)
    strategies_rationnelles = np.where(np.isclose(esperance_gains, gain_max))[0] #Cest Rn dans le papier
    
    mask_valides = np.zeros(M.shape[0], dtype=bool)
    mask_valides[strategies_rationnelles] = True
    matrice_validite = mask_valides[preferences_table]
    indices_meilleurs_choix = np.argmax(matrice_validite, axis=1)
    sigma_n = preferences_table[np.arange(preferences_table.shape[0]), indices_meilleurs_choix]
    
    return sigma_n, strategies_rationnelles


def calculer_moyenne_choix_opti(densite_k, strategies_rationnelles):
    """
    Calcule la distribution des actions jouées par ce niveau d'intelligence.

    Pour chaque type de préférence (k), la fonction sélectionne la meilleure action 
    disponible parmi les stratégies rationnelles (Rn), puis agrège les résultats.

    Args:
        densite_k (array): Proportion de chaque type de préférence dans la population (taille 120).
        strategies_rationnelles (list): Liste des actions valides pour ce tour.

    Returns:
        array: Vecteur de probabilités (taille 5) représentant la fréquence de chaque action.
    """
    mask_actions = np.zeros(5, dtype=bool)
    mask_actions[list(strategies_rationnelles)] = True
    matrix_valid = mask_actions[PREFERENCES_TABLE]
    best_choice_indices = np.argmax(matrix_valid, axis=1)
    
    final_actions = PREFERENCES_TABLE[np.arange(120), best_choice_indices]

    distribution = np.bincount(final_actions, weights=densite_k, minlength=5)

    total_masse = distribution.sum()
    return distribution / total_masse if total_masse > 0 else distribution



def calculate_strat(M, Y):
    """
    Calcul quelle strat est optimale pour tout n et tout k.
    On boucle sur n car le choix des petits n influence le choix des plus gros n
    Args:
        M (np.array): Forme matricielle du jeu.
        Y (np.array): Distribution de la population a l'instant du calcul.
    Returns:
        np.array: Matrice des decisions rationnelles sigma(n,k).
        np.array: Matrice des distributions moyennes mu(n).
    """
    s = np.sum(Y, axis=1)  # repartition demographique par niveau d'intelligence
    s = s[:, np.newaxis] # colonnes redoncantes pour simplifier les calculs
    sigma = np.zeros((N, Nb_preferences2), dtype=int) #cest un choix dans {0,.., 5}
    mu = np.zeros((N, cardE)) #demographie resultante des choix des moins intelligents. Se remplit au fr ur et a mesure
    
    mu[0,:]=calculer_moyenne_choix_opti(Y[0,:], range(cardE))  # niveau 0, choix sans reflexion
    sigma[0,:] = np.array([PREFERENCES_TABLE[k][0] for k in range(Nb_preferences2)])  # choix des niveau 0, toujours la meilleure action selon leur preference
    for n in range(1,N): # on commence par les moisn smart (1) car les 0 reflechissent pas 
        p_inf_n = np.sum(mu[:n]*s[:n], axis=0) / np.sum(s[:n]) #on calcule la strat moyenne des joueurs moins intelligents que n
        sigma[n,:], Rn = calculer_sigma_n(p_inf_n, M, PREFERENCES_TABLE)
        mu[n, :] = calculer_moyenne_choix_opti(Y[n, :], Rn)
    return sigma, mu


def iteration_t(M, Y_t,cout_intelligence, nu):
    """
    Effectue une itération temporelle du modèle de Stahl.

    Args:
        M (np.array): Matrice des gains (5x5).
        Y_t (np.array): Distribution de la population à l'instant t (Nx120).
        cout_intelligence (np.array): Coût associé à chaque niveau d'intelligence.
        nu (float): Taux d'adaptation.

    Returns:
        np.array: Nouvelle distribution de la population à l'instant t+1 (Nx120).
        np.array: Stratégie moyenne à l'instant t (taille 5).
    """
    sigma_t, mu_t = calculate_strat(M, Y_t)
    p_t = np.sum(Y_t, axis=1) @ mu_t  # Calcul de la stratégie moyenne p
    M_barre = np.transpose(p_t) @ M @ p_t  # Gain moyen dans la population
    Y_t_plus_1 = Y_t * ( 1 + nu * ( (M@p_t)[sigma_t] - M_barre - cout_intelligence[:, np.newaxis]) )
    Y_t_plus_1 = np.clip(Y_t_plus_1, a_min=0, a_max=None)  # Évite les valeurs négatives
    somme_Y_t_plus_1 = np.sum(Y_t_plus_1)
    if somme_Y_t_plus_1 > 0:
        return Y_t_plus_1 / somme_Y_t_plus_1, p_t # Normalisation pour que la somme soit 1
    else:
        raise ValueError("La somme de Y_t_plus_1 est nulle, impossible de normaliser.")

def run_simulation(M, Y_0, T, nu, cout_intelligence=np.zeros(N)):
    """
    Exécute la simulation du modèle de Stahl sur T itérations.

    Args:
        M (np.array): Matrice des gains (5x5).
        Y_0 (np.array): Distribution initiale de la population (Nx120).
        nu (float): Taux d'adaptation.
        T (int): Nombre d'itérations temporelles.
        cout_intelligence (np.array): Coût associé à chaque niveau d'intelligence.

    Returns:
        list: Historique des distributions de la population à chaque instant t.

    """
    Y_t = Y_0
    history_Y = [Y_t]
    history_p = []
    for t in range(T):
        Y_t, p_t = iteration_t(M, Y_t, cout_intelligence, nu)
        history_Y.append(Y_t)
        history_p.append(p_t)

    return history_Y, history_p

    
    