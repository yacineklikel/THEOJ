# Stahl's Evolutionary Game Simulation

Implémentation Python du modèle de hiérarchie cognitive ("Smart Players") basé sur l'article de **Dale O. Stahl (1993)**. Ce projet simule la compétition évolutionnaire entre différents niveaux de rationalité ($Smart_0$, $Smart_1$, $Smart_2$, $Smart_3$) dans un jeu symétrique.

## Le Modèle (Jeu 5x5 "L'Escalier vers le Nash")

Pour tester les limites de l'intelligence stratégique, nous utilisons une matrice de gains $G$ spécifique de taille $5 \times 5$. Ce jeu est conçu comme un piège pour distinguer l'intelligence à court terme ($Smart_1$) de l'intelligence à long terme ($Smart_2$).

$$
G = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 \\
2 & 2 & 0 & 0 & 0 \\
3 & 2 & 3 & 1 & 0 \\
\mathbf{4} & \mathbf{4} & \mathbf{4} & \mathbf{3} & 1 \\
0 & 1 & 2 & \mathbf{4} & \mathbf{6}
\end{pmatrix}
$$

### Analyse des Stratégies
* **Stratégies 0, 1, 2 (Le Bruit) :** Rapportent peu. Elles servent de "nourriture" en début de simulation pour les stratégies plus agressives.
* **Stratégie 3 (Le Piège Attractif) :** C'est un optimum local trompeur. Elle rapporte beaucoup (4 points) contre les joueurs faibles. Pour un joueur opportuniste ($Smart_1$) qui ne regarde que l'instant présent, elle apparaît comme la meilleure option.
* **Stratégie 4 (L'Équilibre de Nash) :** La solution optimale cachée. Elle est faible au début (contre les idiots), mais c'est la seule qui bat la Stratégie 3 à long terme (gain de 4 contre 3) et qui offre le gain maximal (6) entre experts.

### Objectif de la Simulation
Nous cherchons à observer un phénomène dynamique en deux temps :
1.  **L'Invasion de l'Imposteur :** La **Stratégie 3** doit d'abord envahir la population en éliminant les stratégies faibles.
2.  **Le Basculement (The Switch) :** Une fois la population stabilisée sur le piège, la **Stratégie 4** doit émerger, "manger" la Stratégie 3 et converger vers l'équilibre de Nash strict.

Ce scénario permet de valider si le niveau d'intelligence $n$ permet réellement d'anticiper l'équilibre final ou si la population reste bloquée sur un optimum local.

## Structure du code:

* `src/stahl_model.py` : Moteur de calcul (simulation du jeu et de levolution de la population)
* `src/simulations.ipynb` : Notebook Jupyter générant les figures du rapport.
