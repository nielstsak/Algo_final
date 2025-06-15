# src/backtesting/config.py

"""
NOTE: Ce fichier est obsolète.

La configuration du backtesting est désormais gérée par le modèle Pydantic `SimulationConfig`
situé dans le module `src.optimization.config`.

Cette centralisation garantit que les mêmes paramètres de simulation (capital, frais,
slippage, levier) sont utilisés de manière cohérente à la fois pour les backtests
simples et pour chaque itération au sein du processus d'optimisation.

Veuillez vous référer à `src.optimization.config.SimulationConfig` pour la structure
de configuration actuelle.
"""

# Ce fichier est intentionnellement laissé vide pour marquer la dépréciation.
# Il pourra être supprimé dans une future version du projet.

pass
