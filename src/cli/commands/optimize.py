import logging
import typer
from pathlib import Path
import os
import pandas as pd

# Configure le chemin pour les imports avant les autres imports du projet
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config as load_yaml_file
from src.core.exceptions import ConfigurationError, DataError, OptimizationError, StrategyLoadError
from src.optimization.optimizer import StrategyOptimizer
from src.optimization.wfo_engine import WFOptimizer
from src.strategies.strategy_loader import StrategyLoader
from src.data.data_manager import DataManager
from src.data.enriched_dataframe import EnrichedDataFrame

# Configuration du logger pour ce module
logger = logging.getLogger(__name__)

# Création d'une application Typer pour la CLI
cli = typer.Typer()


async def optimize_logic(
    strategy_name: str,
    symbol: str,
    config_path: str,
    strategies_config_path: str,
    output_dir: str,
):
    """Contient la logique principale pour l'optimisation de la stratégie."""
    logger.info(
        f"Lancement de l'optimisation pour la stratégie '{strategy_name}' sur '{symbol}'..."
    )

    # 1. Charger la classe de la stratégie
    try:
        strategy_loader = StrategyLoader()
        strategy_class = strategy_loader.get_strategy_class(strategy_name)
        if not strategy_class:
            raise StrategyLoadError(f"La classe pour la stratégie '{strategy_name}' n'a pas pu être chargée.")
        logger.info(f"Classe de stratégie '{strategy_name}' chargée avec succès.")
    except StrategyLoadError as e:
        logger.error(f"Erreur lors du chargement de la stratégie: {e}")
        raise

    # 2. Charger la configuration des paramètres de la stratégie
    strategies_config = load_yaml_file(strategies_config_path)
    if strategy_name not in strategies_config:
        raise ConfigurationError(
            f"Configuration pour la stratégie '{strategy_name}' non trouvée dans '{strategies_config_path}'."
        )
    strategy_params_config = strategies_config[strategy_name].get("optimization_space", {})
    if not strategy_params_config:
        raise ConfigurationError(
            f"La section 'optimization_space' est manquante ou vide pour la stratégie '{strategy_name}' dans '{strategies_config_path}'."
        )
    logger.info("Configuration des paramètres de la stratégie chargée.")

    # 3. Charger les données de marché
    try:
        data_manager = DataManager()
        # MODIFIÉ: Appel de la méthode d'initialisation asynchrone
        await data_manager.initialize()
        
        # Le WFO nécessite des données sur une longue période, typiquement en résolution journalière.
        # L'intervalle exact peut être configuré ailleurs, mais '1d' est un défaut raisonnable pour le WFO.
        # MODIFIÉ: Correction du nom de la méthode et ajout de 'await'
        data_df = await data_manager.get_historical_data(symbol=symbol, timeframe='1d', start_date="2020-01-01") # Exemple de date de début
        if data_df.empty:
            raise DataError(f"Aucune donnée historique trouvée pour le symbole '{symbol}'.")
        enriched_data = EnrichedDataFrame(data_df)
        logger.info(f"{len(data_df)} points de données chargés pour le symbole '{symbol}'.")
    except DataError as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise

    # 4. Initialiser l'optimiseur de stratégie via la méthode de classe `from_yaml`
    try:
        optimizer = StrategyOptimizer.from_yaml(
            config_path=config_path,
            strategy_class=strategy_class,
            strategy_params_config=strategy_params_config,
            data=enriched_data,
            pair_symbol=symbol
        )
    except ConfigurationError as e:
        logger.error(f"Erreur lors de l'initialisation de StrategyOptimizer: {e}")
        raise

    # 5. Exécution du Walk-Forward Optimization (si configuré)
    # Note: La méthode optimizer.run() devra peut-être aussi devenir asynchrone à l'avenir.
    if optimizer.config.wfo_config and optimizer.config.wfo_config.enabled:
        logger.info("Début du processus de Walk-Forward Optimization.")
        wfo_results_df = optimizer.run()

        # 6. Sauvegarde des résultats
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            safe_symbol = symbol.replace('/', '_')
            results_filename = f"wfo_results_{strategy_name}_{safe_symbol}.csv"
            wfo_results_df.to_csv(output_path / results_filename, index=False)
            logger.info(f"Résultats WFO sauvegardés dans : {output_path / results_filename}")
    else:
        logger.info("Début du processus d'optimisation simple.")
        best_trial, study = optimizer.run()
        logger.info(f"Optimisation simple terminée. Meilleur essai: #{best_trial.number} avec la valeur {best_trial.value:.4f}")

    logger.info("Optimisation terminée avec succès.")


@cli.command()
async def optimize(
    strategy_name: str = typer.Option(
        ..., "--strategy-name", "-s", help="Nom de la classe de la stratégie à optimiser."
    ),
    symbol: str = typer.Option(
        ..., "--symbol", "-S", help="Le symbole de la paire à utiliser (ex: 'BTC/USDT')."
    ),
    config_path: str = typer.Option(
        "configs/optimization_config.yaml",
        "--config-path",
        "-c",
        help="Chemin vers le fichier de configuration de l'optimisation.",
    ),
    strategies_config_path: str = typer.Option(
        "configs/strategies_config.yaml",
        "--strategies-config-path",
        help="Chemin vers le fichier de configuration des stratégies.",
    ),
    output_dir: str = typer.Option(
        "optimization_results",
        "--output-dir",
        "-o",
        help="Répertoire où sauvegarder les résultats de l'optimisation.",
    ),
):
    """
    Lance une optimisation pour une stratégie donnée sur un symbole.
    """
    # Initialisation de la configuration globale du logging
    try:
        from src.core.logging_config import setup_logging
        from src.core.config import get_settings
        settings = get_settings()
        setup_logging(settings)
    except Exception as e:
        print(f"Erreur critique lors de la configuration du logging: {e}")
    
    try:
        # MODIFIÉ: Appel de la fonction logique asynchrone avec 'await'
        await optimize_logic(
            strategy_name, symbol, config_path, strategies_config_path, output_dir
        )
    except (ConfigurationError, DataError, OptimizationError, StrategyLoadError) as e:
        logger.error(f"Une erreur contrôlée est survenue: {e}")
        typer.echo(f"ERREUR: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"Une erreur inattendue est survenue: {e}")
        typer.echo(f"ERREUR inattendue: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    cli()
