# src/cli/commands/optimize.py
import asyncio
import click
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Imports du projet ---
from src.optimization.optimizer import StrategyOptimizer
from src.data.data_manager import DataManager
from src.data.enriched_dataframe import EnrichedDataFrame
from src.strategies.strategy_loader import StrategyLoader
from src.core.config import settings
from src.core.exceptions import ConfigurationError, OptimizationError, DataError

logger = logging.getLogger(__name__)

async def async_optimize_logic(strategy: str,
                               symbol_with_slash: str,
                               config_path: str,
                               strategies_config_path: str,
                               output_dir: str):
    """Logique asynchrone pour l'optimisation."""
    click.echo(f"Lancement de l'optimisation pour la stratégie '{strategy}' sur '{symbol_with_slash}'...")
    
    # Prépare le symbole pour la couche de données (sans '/') et pour l'optimiseur (avec '/').
    symbol_for_data = symbol_with_slash.replace('/', '')
    
    # --- 1. Chargement des configurations ---
    with open(strategies_config_path, 'r') as f:
        strategies_config = yaml.safe_load(f)
    
    strategy_params_config = strategies_config.get('strategies', {}).get(strategy)
    if not strategy_params_config:
        raise ConfigurationError(f"Configuration pour la stratégie '{strategy}' non trouvée dans '{strategies_config_path}'.")

    # --- 2. Chargement de la stratégie ---
    strategy_loader = StrategyLoader()
    strategy_class = strategy_loader.get_strategy_class(strategy)
    if not strategy_class:
        raise ConfigurationError(f"Classe de stratégie '{strategy}' non trouvée par le loader.")
    logger.info(f"Classe de la stratégie '{strategy}' chargée avec succès.")

    # --- 3. Chargement des données ---
    async with DataManager() as data_manager:
        # Utilise le symbole sans la barre oblique pour charger les données
        enriched_data = await data_manager.get_enriched_klines(pair=symbol_for_data)
        if enriched_data is None:
            raise DataError(f"Impossible de charger les données pour le symbole {symbol_with_slash}.")
        
        logger.info(f"Données chargées pour {symbol_with_slash}. Total de {len(enriched_data.df)} bougies.")

    # --- 4. Initialisation de l'optimiseur ---
    # L'optimiseur utilise le symbole original avec la barre oblique
    optimizer = StrategyOptimizer.from_yaml(
        config_path=config_path,
        strategy_class=strategy_class,
        strategy_params_config=strategy_params_config,
        data=enriched_data,
        pair_symbol=symbol_with_slash
    )

    # --- 5. Lancement de l'optimisation ---
    results = optimizer.run()

    # --- 6. Traitement et sauvegarde des résultats ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if isinstance(results, pd.DataFrame):
        click.echo("Optimisation Walk-Forward terminée.")
        click.echo(f"Nombre de validations OOS : {len(results)}")
        output_file = output_path / f"wfo_results_{strategy}_{symbol_for_data}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(output_file, index=False)
        click.echo(f"Résultats détaillés du WFO sauvegardés dans : {output_file}")
    else:
        best_trial, study = results
        click.echo("Optimisation simple terminée.")
        click.secho("\n--- MEILLEUR ESSAI TROUVÉ ---", fg="green", bold=True)
        click.echo(f"  Numéro de l'essai: {best_trial.number}")
        
        if optimizer.is_multi_objective:
            for i, obj in enumerate(optimizer.config.optuna_config.objectives):
                click.echo(f"  Objectif '{obj.name}': {best_trial.values[i]:.4f}")
        else:
            metric_name = optimizer.config.optuna_config.objectives[0].name
            click.echo(f"  Objectif '{metric_name}': {best_trial.value:.4f}")
        
        click.echo("  Meilleurs paramètres:")
        for key, value in best_trial.params.items():
            click.echo(f"    {key}: {value}")

        study_results_df = study.trials_dataframe()
        output_file = output_path / f"study_results_{strategy}_{symbol_for_data}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        study_results_df.to_csv(output_file, index=False)
        click.echo(f"\nRésultats complets de l'étude sauvegardés dans : {output_file}")

@click.command()
@click.option('--strategy-name', '-s', required=True, help="Nom de la classe de la stratégie à optimiser.")
@click.option('--symbol', required=True, help="Symbole à utiliser pour les données (ex: 'WIF/USDC').")
@click.option('--config-path', '-c', type=click.Path(exists=True, dir_okay=False, readable=True),
              default='configs/optimization_config.yaml', help="Chemin vers le fichier de configuration de l'optimisation.")
@click.option('--strategies-config-path', type=click.Path(exists=True, dir_okay=False, readable=True),
              default='configs/strategies_config.yaml', help="Chemin vers le fichier de configuration des stratégies.")
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, writable=True),
              default='optimization_results', help="Répertoire de sortie pour les résultats.")
def optimize(strategy_name: str,
             symbol: str,
             config_path: str,
             strategies_config_path: str,
             output_dir: str):
    """
    Lance une session d'optimisation d'hyperparamètres pour une stratégie donnée,
    en utilisant la configuration fournie.
    """
    try:
        # Passe le symbole avec la barre oblique à la fonction de logique
        asyncio.run(async_optimize_logic(strategy_name, symbol, config_path, strategies_config_path, output_dir))
    except (ConfigurationError, OptimizationError, FileNotFoundError, DataError) as e:
        logger.error(f"Une erreur de configuration ou d'optimisation est survenue: {e}", exc_info=True)
        click.secho(f"ERREUR: {e}", fg="red")
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue: {e}", exc_info=True)
        click.secho(f"ERREUR inattendue: {e}", fg="red")

if __name__ == '__main__':
    optimize()
