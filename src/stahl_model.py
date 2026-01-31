import numpy as np

class Game:
    """
    Définit le jeu symétrique 3x3.
    Je l'ai mise sous forme d'une classe pour faire propre mais en vrai cest jjuste la matrice du jeu
    sous la forme d'un array numpy avec une method pour acceder a l'elelement [i,j].
    """
    def __init__(self):
        self.payoff_matrix = np.array([
            [0, 1, 0], 
            [1, 2, 0],
            [2, 2, 4]
        ])
        
    def get_payoff(self, strategy_p1, strategy_p2):
        return self.payoff_matrix[strategy_p1, strategy_p2]

class Population:
    """
    Gère la dynamique de la population et les différents types de joueurs.
    """
    def __init__(self, game):
        self.game = game
        self.types = [] # Liste des objets PlayerType
        self.proportions = [] # Liste des proportions actuelles (somme = 1)
        
    def add_type(self, player_type, initial_proportion):
        self.types.append(player_type)
        self.proportions.append(initial_proportion)
        
    def normalize_proportions(self):
        """Assure que la somme des proportions = 1"""
        total = sum(self.proportions)
        self.proportions = [p / total for p in self.proportions]

    def get_population_strategy_distribution(self):
        """
        Calcule la probabilité globale qu'une stratégie (0, 1 ou 2) soit jouée
        par l'ensemble de la population actuelle.
        Retourne un vecteur [prob_S0, prob_S1, prob_S2]
        """
        n_strategies = self.game.payoff_matrix.shape[0]
        distribution = np.zeros(n_strategies)
        
        for idx, p_type in enumerate(self.types):
            # Récupère la probabilité que ce type joue chaque stratégie
            # Note: Pour Smart_0 c'est fixe, pour Smart_n ça dépend de ses calculs
            strat_probs = p_type.get_strategy_distribution(self.proportions, self.types)
            distribution += self.proportions[idx] * strat_probs
            
        return distribution

    def step_evolution(self, dt=0.1):
        """
        Exécute un pas de temps de la dynamique de réplication (Eq 4 de Stahl).
        """
        current_proportions = np.array(self.proportions)
        n_types = len(self.types)
        fitnesses = np.zeros(n_types)
        
        # 1. Calculer la distribution globale des stratégies jouées dans la population
        pop_distribution = self.get_population_strategy_distribution()
        
        # 2. Calculer le Fitness (Gain espéré) pour chaque TYPE de joueur
        for i, p_type in enumerate(self.types):
            # Le gain dépend de sa stratégie face à la distribution de la population
            # Moins le coût cognitif
            my_strat_dist = p_type.get_strategy_distribution(self.proportions, self.types)
            
            # Gain brut = Somme(Ma_prob_jouer_S * Prob_Autre_jouer_S' * Gain(S, S'))
            gross_payoff = 0
            for s_mine in range(3):
                for s_other in range(3):
                    gross_payoff += (my_strat_dist[s_mine] * pop_distribution[s_other] * self.game.payoff_matrix[s_mine, s_other])
            
            fitnesses[i] = gross_payoff - p_type.cost

        # 3. Calculer le Fitness moyen de la population
        avg_fitness = np.dot(current_proportions, fitnesses)
        
        # 4. Mise à jour des proportions (Équation de réplication)
        # dy/dt = y * (fitness - avg_fitness)
        new_proportions = np.zeros(n_types)
        for i in range(n_types):
            change = current_proportions[i] * (fitnesses[i] - avg_fitness)
            new_proportions[i] = current_proportions[i] + change * dt
            
        # Sécurité pour éviter les valeurs négatives ou > 1 dues à l'approximation discrète
        new_proportions = np.clip(new_proportions, 0.0001, 1.0) 
        
        # Mise à jour et normalisation
        self.proportions = list(new_proportions)
        self.normalize_proportions()
        
        return self.proportions

class PlayerType:
    def __init__(self, name, level, cost=0, fixed_strategy=None):
        self.name = name
        self.level = level # 0, 1, 2...
        self.cost = cost
        self.fixed_strategy = fixed_strategy # Uniquement pour Smart_0
        
    def get_strategy_distribution(self, population_shares, player_types):
        """
        Retourne un vecteur [p0, p1, p2] indiquant la probabilité de jouer chaque coup.
        """
        # --- NIVEAU 0 : Joue une stratégie fixe ---
        if self.level == 0:
            dist = np.zeros(3)
            dist[self.fixed_strategy] = 1.0
            return dist
            
        # --- NIVEAU 1 : Best Response à la distribution visible ---
        # Stahl suppose que Smart_n croit que tout le monde est < n.
        # Pour simplifier ici (modèle standard), Smart_1 réagit à la population réelle observée.
        
        # 1. Estimer ce que jouent les autres
        # Pour Smart_1, on regarde la distribution agrégée des stratégies
        total_dist = np.zeros(3)
        # On recalcule la distribution globale (approximation simplifiée pour éviter récursion infinie)
        # Dans un modèle pur Stahl, Smart_n utilise une croyance Bayesienne sur les types n-1.
        # Ici, version "Best Response Dynamics" classique pour vulgarisation.
        
        if self.level >= 1:
            # Calcul des gains espérés pour chaque coup (0, 1, 2)
            expected_payoffs = np.zeros(3)
            
            # Reconstruction de la perception de la population
            # (Pour simplifier le code, on passe la vraie distribution ici)
            # Dans une implémentation stricte, il faudrait passer l'objet Game et recalculer
            pass 
            
        # NOTE : Pour garder le code simple et fonctionnel pour le mémoire,
        # nous allons coder "en dur" la logique Best Response pour ce jeu 3x3 spécifique
        # car calculer dynamiquement la récursion Smart_n est complexe en Python pur.
        
        return self._compute_best_response_logic(population_shares, player_types)

    def _compute_best_response_logic(self, shares, types):
        """Logique de décision simplifiée pour la simulation"""
        # On calcule ce que joue la population en moyenne
        pop_play = np.zeros(3)
        for idx, t in enumerate(types):
            # Pour éviter la récursion infinie, on considère que les autres Smart 
            # jouent leur stratégie "préférée" ou aléatoire à l'instant t-1
            # Ici on triche un peu pour la stabilité : on regarde les Smart_0
            if t.level == 0:
                pop_play[t.fixed_strategy] += shares[idx]
            else:
                # On assume que les autres smarts jouent uniformément si on ne sait pas
                pop_play += (shares[idx] / 3.0) 
        
        # Matrice G
        # [0, 1, 0]
        # [1, 2, 0]
        # [2, 2, 4]
        
        # Gain espéré de jouer 0: 0*P0 + 1*P1 + 0*P2
        # Gain espéré de jouer 1: 1*P0 + 2*P1 + 0*P2
        # Gain espéré de jouer 2: 2*P0 + 2*P1 + 4*P2
        
        gains = [
            0*pop_play[0] + 1*pop_play[1] + 0*pop_play[2],
            1*pop_play[0] + 2*pop_play[1] + 0*pop_play[2],
            2*pop_play[0] + 2*pop_play[1] + 4*pop_play[2]
        ]
        
        # Si je suis Smart_2, j'anticipe que les Smart_1 vont jouer la Best Response
        if self.level == 2:
            # Smart 2 pense que les autres vont jouer le MAX des gains ci-dessus
            best_strat_others = np.argmax(gains)
            # Ma réponse est la meilleure réponse à best_strat_others
            # Si autres jouent 0 -> Je joue 2 (gain 2)
            # Si autres jouent 1 -> Je joue 1 ou 2 (gain 2) -> disons 2 pour Nash
            # Si autres jouent 2 -> Je joue 2 (gain 4)
            # Dans cette matrice spécifique, 2 est dominant ou Nash presque partout.
            final_strat = 2 
        else:
            # Smart 1 joue juste le max immédiat
            final_strat = np.argmax(gains)
            
        dist = np.zeros(3)
        dist[final_strat] = 1.0
        return dist