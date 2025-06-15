# src/cli/commands/download.py
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any
import hashlib

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.core.config import get_settings
from src.core.constants import Kline, System
from src.core.exceptions import (
    ConfigurationError,
    DataDownloadError,
    StorageError,
    AlgoBotException,
)
from src.data.data_manager import DataManager

# --- Gestion des checkpoints ---
CHECKPOINT_DIR = get_settings().data.storage_path / ".cli_checkpoints"

def get_checkpoint_path(pairs: List[str], interval: str, start_date: Optional[datetime]) -> Path:
    """Génère un nom de fichier de checkpoint basé sur les paramètres de la session."""
    pairs_hash = hashlib.sha1(','.join(sorted(pairs)).encode()).hexdigest()[:8]
    start_date_str = start_date.strftime('%Y%m%d') if start_date else 'all'
    return CHECKPOINT_DIR / f"download_{pairs_hash}_{interval}_{start_date_str}.json"

def save_checkpoint(path: Path, state: Dict[str, Any]):
    """Sauvegarde l'état du téléchargement dans un fichier de checkpoint."""
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        state['last_saved_at'] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug(f"Checkpoint saved to {path}")
    except IOError as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    """Charge l'état du téléchargement depuis un fichier de checkpoint."""
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load checkpoint, starting fresh: {e}")
    return None

def clear_checkpoint(path: Path):
    """Supprime un fichier de checkpoint."""
    try:
        if path.exists():
            path.unlink()
            logger.info(f"Checkpoint {path} cleared.")
    except IOError as e:
        logger.error(f"Failed to clear checkpoint {path}: {e}")

# --- Validation de date ---
def validate_date(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[datetime]:
    """Valide le format de date (YYYY-MM-DD) et le convertit en datetime UTC."""
    if value is None:
        return None
    try:
        dt_naive = datetime.strptime(value, "%Y-%m-%d")
        return dt_naive.replace(tzinfo=timezone.utc)
    except ValueError:
        raise click.BadParameter(f"Date format for '{param.name}' must be YYYY-MM-DD. Got '{value}'")

# --- Logique de téléchargement asynchrone ---
async def async_download_logic(
    pairs: List[str],
    start_date: Optional[datetime],
    interval: str,
    update: bool,
    resume: bool
):
    """Gère la logique principale du téléchargement de manière asynchrone."""
    checkpoint_file_path = get_checkpoint_path(pairs, interval, start_date)
    
    completed_pairs: List[str] = []
    if resume:
        checkpoint_data = load_checkpoint(checkpoint_file_path)
        if checkpoint_data:
            completed_pairs = checkpoint_data.get("completed_pairs", [])
            logger.info(f"Resuming download. Already completed pairs: {completed_pairs}")

    pairs_to_process = [p for p in pairs if p not in completed_pairs]
    if not pairs_to_process:
        logger.success("All specified pairs are already downloaded according to checkpoint.")
        clear_checkpoint(checkpoint_file_path)
        return

    async with DataManager() as data_manager:
        with tqdm(total=len(pairs), initial=len(completed_pairs), desc="Overall Progress", unit="pair") as pbar:
            for pair in pairs_to_process:
                pbar.set_description(f"Processing {pair}")
                effective_start_date = start_date
                
                if update:
                    logger.info(f"Update mode for {pair}: checking for last stored data...")
                    data_range = await data_manager.get_data_range(pair, interval)
                    if data_range and data_range[1]:
                        # Commence à télécharger à partir de la kline suivant la dernière stockée
                        effective_start_date = data_range[1] + pd.to_timedelta(1, unit='ms')
                        logger.info(f"Updating from {effective_start_date} for {pair}.")

                if not effective_start_date:
                    # Pour un téléchargement initial sans date de début, Binance utilise la première date disponible
                    logger.info(f"No start date specified for {pair}. Fetching all available history.")
                    # Binance nécessite une date de début pour l'historique lointain
                    effective_start_date = datetime(2017, 1, 1, tzinfo=timezone.utc)

                try:
                    result = await data_manager.download_historical_data(
                        pairs=[pair],
                        start_date=effective_start_date,
                        interval=interval
                    )
                    
                    if not result.get("errors"):
                        completed_pairs.append(pair)
                        save_checkpoint(checkpoint_file_path, {"completed_pairs": completed_pairs})
                    else:
                        logger.error(f"Error processing {pair}: {result['errors'][0]['error']}")

                except Exception as e:
                    logger.error(f"A critical error occurred while processing {pair}: {e}", exc_info=True)

                pbar.update(1)

    logger.success("Download process completed.")
    clear_checkpoint(checkpoint_file_path)

# --- Commande CLI ---
@click.command("download", help="Download historical kline data from Binance.")
@click.option(
    "--pairs", "-p", "pairs_str", required=True, type=str,
    help="Comma-separated list of trading pairs (e.g., BTCUSDC,ETHUSDC)."
)
@click.option(
    "--start-date", "-s", callback=validate_date, type=str,
    help="Start date for historical data in YYYY-MM-DD format. Defaults to Binance's earliest data."
)
@click.option(
    "--update", "-u", is_flag=True, default=False,
    help="Update mode: Download only new data since the last stored kline. Overrides --start-date."
)
@click.option(
    "--resume", "-r", is_flag=True, default=False,
    help="Resume download from the last checkpoint if available."
)
@click.option(
    "--interval", "-i",
    type=click.Choice([getattr(Kline, attr) for attr in dir(Kline) if attr.startswith('INTERVAL_')]),
    default=Kline.INTERVAL_1MINUTE, show_default=True,
    help="Kline interval."
)
def download_cli(
    pairs_str: str,
    start_date: Optional[datetime],
    update: bool,
    resume: bool,
    interval: str,
):
    """Point d'entrée de la CLI pour le téléchargement de données."""
    try:
        parsed_pairs = [pair.strip().upper() for pair in pairs_str.split(",") if pair.strip()]
        if not parsed_pairs:
            raise click.UsageError("Pairs list cannot be empty.")

        logger.info("Starting historical data download process...")
        asyncio.run(async_download_logic(parsed_pairs, start_date, interval, update, resume))

    except (click.UsageError, click.BadParameter) as e:
        logger.error(f"CLI Error: {e.message}")
        sys.exit(e.exit_code)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Download command finished.")

if __name__ == "__main__":
    download_cli()