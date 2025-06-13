# Projet de Bot de Trading Algorithmique

> **Note :** Ce projet est développé dans un but purement pédagogique et d'apprentissage personnel. Il n'est pas destiné à être utilisé pour du trading avec de l'argent réel et ne constitue en aucun cas un conseil en investissement.

## 1. Description Générale

Ce projet fournit une infrastructure complète pour développer et évaluer des stratégies de trading quantitatives. Il est conçu autour de modules distincts pour la gestion des données, la définition des stratégies, le backtesting, l'optimisation des hyperparamètres et l'exécution.

L'architecture permet de tester rapidement de nouvelles idées de stratégies et de les optimiser rigoureusement grâce à des techniques avancées comme le Walk-Forward Optimization (WFO) et les algorithmes génétiques (NSGA-II).

## 2. Fonctionnalités Principales

* **Gestion des Données :**
    * Connexion à l'API **Binance** pour le téléchargement des données de marché (K-lines).
    * Stockage de données performant via **Parquet** ou **PostgreSQL**.
    * Système de cache avec **Redis** pour accélérer l'accès aux données.
    * Traitement et enrichissement des données pour supporter des analyses multi-timeframe.

* **Système de Stratégies Modulaire :**
    * Classe de base (`BaseStrategy`) pour standardiser la création de nouvelles stratégies.
    * Configuration déclarative des paramètres (fixes et optimisables) via des modèles **Pydantic**.
    * Chargement dynamique des stratégies pour une extensibilité maximale.

* **Moteur de Backtesting Avancé :**
    * Utilisation de **`vectorbt`** pour des simulations rapides et vectorisées.
    * Modélisation personnalisable des **frais de transaction** et du **slippage** (glissement de prix).
    * Intégration d'une condition de **"stop-ruine"** pour interrompre les backtests non viables et accélérer l'optimisation.

* **Optimisation d'Hyperparamètres :**
    * Intégration complète avec **`Optuna`** pour l'optimisation des paramètres de stratégie.
    * Support du **multi-objectif** avec l'échantillonneur `NSGAIISampler`.
    * Implémentation d'un moteur de **Walk-Forward Optimization (WFO)** pour valider la robustesse des stratégies dans le temps, incluant le purging et l'embargo des données.

* **Interface en Ligne de Commande (CLI) :**
    * Commandes claires basées sur **`click`** pour gérer le cycle de vie du bot :
        * `download`: Télécharger les données historiques.
        * `backtest`: Lancer un backtest simple d'une stratégie.
        * `optimize`: Lancer une session d'optimisation complète (simple ou WFO).

## 3. Structure du Projet


.
├── configs/                # Fichiers de configuration YAML.
│   ├── config.yaml         # Configuration générale (API, DB, etc.).
│   ├── optimization_config.yaml # Paramètres pour Optuna et WFO.
│   └── strategies_config.yaml # Plages de paramètres pour chaque stratégie.
│
├── data/                   # Données brutes et traitées (ex: fichiers Parquet).
│
├── logs/                   # Fichiers de logs générés par l'application.
│
├── optimization_results/   # CSV et rapports générés par les optimisations.
│
├── src/                    # Code source de l'application.
│   ├── backtesting/        # Moteur de backtesting (basé sur vectorbt).
│   ├── cli/                # Commandes de l'interface en ligne de commande.
│   ├── core/               # Composants principaux (configuration, exceptions).
│   ├── data/               # Gestion des données (clients API, stockage, cache).
│   ├── optimization/       # Logique d'optimisation (Optuna, WFO).
│   └── strategies/         # Définition des stratégies de trading.
│
└── requirements.txt        # Dépendances Python du projet.


## 4. Installation

1.  **Cloner le dépôt :**
    ```bash
    git clone <URL_DU_DEPOT>
    cd <NOM_DU_DEPOT>
    ```

2.  **Créer et activer un environnement virtuel :**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
