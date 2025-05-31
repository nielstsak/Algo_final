# main_data.py
import asyncio
from datetime import datetime
from loguru import logger

from src.core.config import settings
from src.core.constants import Kline
from src.core.logging_config import setup_logging
from src.data.data_manager import DataManager

def simple_progress_callback(stats: dict):
    """
    Callback simple pour afficher la progression du téléchargement.
    """
    if stats.get("total_pairs", 0) == 0:
        logger.info("Calcul de la progression...")
        return

    progress_percentage = stats.get("progress_percentage", 0)
    current_pair_progress = stats.get("current_pair_progress", 0)
    current_pair_total = stats.get("current_pair_total", 0)
    current_pair_name = stats.get("current_pair", "N/A")
    
    if current_pair_total > 0:
        pair_progress_percentage = (current_pair_progress / current_pair_total) * 100
        logger.info(
            f"Progression Globale: {progress_percentage:.2f}% | "
            f"Paire actuelle: {current_pair_name} ({current_pair_progress}/{current_pair_total} klines, {pair_progress_percentage:.2f}%)"
        )
    else:
        logger.info(
             f"Progression Globale: {progress_percentage:.2f}% | "
             f"Paire actuelle: {current_pair_name} (Initialisation...)"
        )

async def main():
    """
    Fonction principale pour télécharger les données historiques.
    """
    # 1. Configuration du logging
    # Note: setup_logging est appelé au chargement de src.core.config via settings.
    # Si vous avez besoin de reconfigurer ou d'assurer sa configuration ici:
    # setup_logging(settings) # Assurez-vous que `settings` est bien l'instance chargée.
    logger.info("Initialisation du script de téléchargement de données.")

    # 2. Récupérer les paires depuis la configuration
    pairs_to_download = settings.trading.allowed_pairs
    if not pairs_to_download:
        logger.error("Aucune paire de trading n'est configurée dans 'settings.trading.allowed_pairs'. Arrêt.")
        return

    logger.info(f"Paires à télécharger : {pairs_to_download}")

    # 3. Définir la période de téléchargement
    # Ces dates peuvent être rendues configurables (ex: via arguments CLI ou une section dédiée dans config.yaml)
    # Utilisation des dates d'exemple du Cahier des Spécifications Fonctionnelles [cite: 2]
    start_date_str = "2024-10-01"
    end_date_str = "2025-05-30" # Fin de journée, donc inclusif jusqu'à la fin de cette date.

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        # Pour que end_date soit inclusif jusqu'à la fin de la journée
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    except ValueError as e:
        logger.error(f"Format de date invalide. Utilisez YYYY-MM-DD. Erreur : {e}")
        return

    logger.info(f"Période de téléchargement : de {start_date_str} à {end_date_str}")

    # 4. Initialiser et utiliser le DataManager
    data_manager = DataManager()
    try:
        await data_manager.initialize()
        logger.info("DataManager initialisé.")

        download_stats = await data_manager.download_historical_data(
            pairs=pairs_to_download,
            start_date=start_date,
            end_date=end_date,
            interval=Kline.INTERVAL_1MINUTE, # Téléchargement des données de base à 1 minute
            resume=True, # Permet de reprendre un téléchargement interrompu
            progress_callback=simple_progress_callback
        )

        logger.info("Statistiques de téléchargement :")
        for key, value in download_stats.items():
            if key == "errors" and isinstance(value, list) and value:
                logger.warning(f"  {key}:")
                for err_info in value:
                    logger.warning(f"    - Paire: {err_info.get('pair')}, Erreur: {err_info.get('error')}")
            else:
                logger.info(f"  {key}: {value}")

        if download_stats.get("success", False):
            logger.success("Téléchargement des données historiques terminé avec succès.")
        else:
            logger.warning("Le téléchargement des données historiques s'est terminé avec des erreurs ou incomplètement.")

    except Exception as e:
        logger.exception(f"Une erreur critique est survenue lors du processus de téléchargement : {e}")
    finally:
        logger.info("Fermeture du DataManager...")
        await data_manager.close()
        logger.info("DataManager fermé.")

if __name__ == "__main__":
    # Configuration initiale du logging si non faite via l'import de settings
    # Ceci assure que les logs sont configurés avant toute autre opération.
    # Si `src.core.config.settings` est importé, `setup_logging` est déjà appelé.
    # Sinon, il faudrait l'appeler explicitement ici.
    # from src.core.config import settings # Assure le chargement et la config du log
    # from src.core.logging_config import setup_logging
    # setup_logging(settings)
    
    asyncio.run(main())