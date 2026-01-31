# Stahl's Evolutionary Game Simulation

Implémentation Python du modèle de hiérarchie cognitive ("Smart Players") basé sur l'article de **Dale O. Stahl (1993)**. Ce projet simule la compétition évolutionnaire entre différents niveaux de rationalité ($Smart_0$, $Smart_1$, $Smart_2$) dans un jeu symétrique.

## Le Modèle (Jeu 3x3)

La simulation repose sur une matrice de gains $G$ à **3 états** (Stratégies 0, 1, 2) configurée pour illustrer la convergence :

$$
G = \begin{pmatrix}
0 & 1 & 0 \\
1 & 2 & 0 \\
2 & 2 & 4
\end{pmatrix}
$$

* **Stratégie 0 (Dominée) :** Faible gain, destinée à disparaître rapidement.
* **Stratégie 1 (Piège local) :** Offre un gain moyen (2), mais n'est pas optimale à long terme.
* **Stratégie 2 (Équilibre de Nash) :** L'équilibre strict et optimal du jeu (Gain max de 4).

L'objectif est d'observer si l'intelligence ($Smart_n$) permet d'atteindre cet équilibre plus vite que le hasard.

## Structure du dépot

* `src/stahl_engine.py` : Moteur de calcul (Classes `Game`, `PlayerType`, `Population`).
* `src/simulations.ipynb` : Notebook Jupyter générant les 4 figures du rapport.
* `src/figures/` : Dossier de sauvegarde automatique des graphiques (.png).
* `report/` : Contient le mémoire final (PDF).
