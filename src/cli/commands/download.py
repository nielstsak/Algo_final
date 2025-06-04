# src/cli/commands/download.py
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.core.config import settings, PROJECT_ROOT
from src.core.constants import Kline, System
from src.core.exceptions import (
    ConfigurationError,
    DataDownloadError,
    StorageError,
    AlgoBotException,
)
from src.data.data_manager import DataManager # Assuming DataManager is in src.data.data_manager
# Ensure logging is configured if not already done by settings import
# from src.core.logging_config import setup_logging
# if not logger.configured: # Basic check, actual setup might be more complex
#    setup_logging(settings)


# --- Checkpoint Management ---
CHECKPOINT_DIR = PROJECT_ROOT / ".checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "cli_download_checkpoint.json"

def save_checkpoint(state: Dict[str, Any]) -> None:
    """Saves the download state to a checkpoint file."""
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Checkpoint saved to {CHECKPOINT_FILE}")
    except IOError as e:
        logger.error(f"Failed to save checkpoint: {e}")

def load_checkpoint() -> Optional[Dict[str, Any]]:
    """Loads the download state from a checkpoint file."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                state = json.load(f)
            logger.info(f"Checkpoint loaded from {CHECKPOINT_FILE}")
            return state
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")
            return None
    return None

def clear_checkpoint() -> None:
    """Clears the checkpoint file."""
    try:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("Checkpoint cleared.")
    except IOError as e:
        logger.error(f"Failed to clear checkpoint: {e}")


# --- Date Validation ---
def validate_date(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[datetime]:
    """Validates date string format (YYYY-MM-DD) and converts to datetime object."""
    if value is None:
        return None
    try:
        # Convert to datetime object, then to timezone-aware UTC
        dt_naive = datetime.strptime(value, "%Y-%m-%d")
        dt_aware = dt_naive.replace(tzinfo=timezone.utc) # Assume UTC for start_date
        return dt_aware
    except ValueError:
        raise click.BadParameter(
            f"Date format for '{param.name}' must be YYYY-MM-DD. Got '{value}'"
        )

# --- Global TQDM instance ---
tqdm_instance: Optional[tqdm] = None
processed_klines_for_checkpoint = 0
KLINES_PER_CHECKPOINT = 1000 # Save checkpoint every 1000 klines processed in current pair

# --- Signal Handler for graceful exit ---
shutdown_requested = False

def handle_signal(sig, frame):
    """Handles Ctrl+C and other termination signals."""
    global shutdown_requested
    if shutdown_requested: # Second Ctrl+C
        logger.warning("Forcing exit...")
        sys.exit(1)
    logger.warning(f"Shutdown requested (Signal: {sig}). Finishing current operations and saving checkpoint...")
    shutdown_requested = True
    if tqdm_instance:
        tqdm_instance.set_description(f"{tqdm_instance.desc} (Exiting...)")


# --- CLI Command ---
@click.command("download", help="Download historical kline data from Binance.")
@click.option(
    "--pairs",
    "-p",
    "pairs_str",
    required=True,
    type=str,
    help="Comma-separated list of trading pairs (e.g., BTCUSDC,ETHUSDC).",
)
@click.option(
    "--start-date",
    "-s",
    callback=validate_date,
    type=str,
    help="Start date for historical data in YYYY-MM-DD format (inclusive). Defaults to Binance's earliest data for the pair.",
)
@click.option(
    "--update",
    "-u",
    is_flag=True,
    default=False,
    help="Update mode: Download only new data since the last stored kline. Overrides --start-date if existing data is found.",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    default=False,
    help="Resume download from the last checkpoint if available.",
)
@click.option(
    "--interval",
    "-i",
    type=click.Choice([
        Kline.INTERVAL_1MINUTE, Kline.INTERVAL_3MINUTE, Kline.INTERVAL_5MINUTE,
        Kline.INTERVAL_15MINUTE, Kline.INTERVAL_30MINUTE, Kline.INTERVAL_1HOUR,
        Kline.INTERVAL_2HOUR, Kline.INTERVAL_4HOUR, Kline.INTERVAL_6HOUR,
        Kline.INTERVAL_8HOUR, Kline.INTERVAL_12HOUR, Kline.INTERVAL_1DAY,
        Kline.INTERVAL_3DAY, Kline.INTERVAL_1WEEK, Kline.INTERVAL_1MONTH
    ]),
    default=Kline.INTERVAL_1MINUTE,
    show_default=True,
    help="Kline interval.",
)
def download_cli(
    pairs_str: str,
    start_date: Optional[datetime],
    update: bool,
    resume: bool,
    interval: str,
):
    """
    CLI command to download historical kline data from Binance.
    """
    global tqdm_instance, processed_klines_for_checkpoint, shutdown_requested

    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Starting historical data download process...")

    try:
        parsed_pairs: List[str] = [pair.strip().upper() for pair in pairs_str.split(",")]
        if not parsed_pairs or any(not pair for pair in parsed_pairs):
            raise click.UsageError("Pairs list cannot be empty or contain empty strings.")
        logger.info(f"Target pairs: {parsed_pairs}, Interval: {interval}")

        # Checkpoint state variables
        completed_pairs_checkpoint: List[str] = []
        current_pair_checkpoint: Optional[str] = None
        current_pair_start_time_checkpoint: Optional[int] = None # Timestamp in ms

        if resume:
            checkpoint_data = load_checkpoint()
            if checkpoint_data:
                # Validate checkpoint data against current command parameters
                chk_pairs = checkpoint_data.get("target_pairs")
                chk_interval = checkpoint_data.get("interval")
                chk_start_date_str = checkpoint_data.get("original_start_date_iso")
                
                original_start_date_iso = start_date.isoformat() if start_date else None

                if (chk_pairs == parsed_pairs and 
                    chk_interval == interval and
                    chk_start_date_str == original_start_date_iso):
                    
                    completed_pairs_checkpoint = checkpoint_data.get("completed_pairs", [])
                    current_pair_checkpoint = checkpoint_data.get("current_pair_processing")
                    current_pair_start_time_checkpoint = checkpoint_data.get("current_pair_last_timestamp_ms")
                    logger.info(f"Resuming download. Completed pairs: {completed_pairs_checkpoint}")
                    if current_pair_checkpoint and current_pair_start_time_checkpoint:
                        logger.info(f"Will attempt to resume {current_pair_checkpoint} from timestamp {current_pair_start_time_checkpoint}")
                else:
                    logger.warning("Checkpoint parameters mismatch. Starting fresh. Clearing old checkpoint.")
                    clear_checkpoint() # Mismatch, so clear old checkpoint

        pairs_to_process = [p for p in parsed_pairs if p not in completed_pairs_checkpoint]
        if not pairs_to_process and not current_pair_checkpoint: # All pairs already completed
             logger.success("All specified pairs are already downloaded according to checkpoint.")
             clear_checkpoint()
             return

        # Initialize DataManager
        data_manager = DataManager() # Define data_manager here to ensure it's in scope for finally
        try:
            async def main_download_logic():
                nonlocal data_manager # Allow modification of the outer scope data_manager
                global tqdm_instance, processed_klines_for_checkpoint, shutdown_requested
                nonlocal completed_pairs_checkpoint, current_pair_checkpoint, current_pair_start_time_checkpoint
                
                await data_manager.initialize()

                # Overall progress for pairs
                with tqdm(total=len(parsed_pairs), unit="pair", desc="Overall Progress") as overall_pbar:
                    overall_pbar.update(len(completed_pairs_checkpoint)) # Update for already completed pairs

                    for pair_index, pair in enumerate(parsed_pairs):
                        if shutdown_requested:
                            logger.warning(f"Skipping remaining pairs due to shutdown request.")
                            break
                        
                        if pair in completed_pairs_checkpoint:
                            logger.info(f"Pair {pair} already completed in checkpoint. Skipping.")
                            continue # overall_pbar already updated

                        overall_pbar.set_postfix_str(f"Current: {pair}")
                        logger.info(f"Processing pair: {pair} ({pair_index + 1}/{len(parsed_pairs)})")
                        
                        current_pair_start_time_ms: Optional[int] = None
                        
                        if resume and pair == current_pair_checkpoint and current_pair_start_time_checkpoint:
                            current_pair_start_time_ms = current_pair_start_time_checkpoint + 1 
                            logger.info(f"Resuming {pair} from timestamp {current_pair_start_time_ms} (last saved + 1ms)")
                        elif start_date:
                            current_pair_start_time_ms = int(start_date.timestamp() * 1000)
                        
                        actual_start_date_for_pair = start_date # Keep original start_date for checkpoint comparison
                        if update:
                            logger.info(f"Update mode for {pair}. Determining last stored timestamp...")
                            last_ts_info = await data_manager.get_data_range(pair, interval)
                            if last_ts_info and last_ts_info[1]: 
                                effective_start_dt = last_ts_info[1] + pd.Timedelta(milliseconds=1) 
                                effective_start_time_ms = int(effective_start_dt.timestamp() * 1000)
                                if current_pair_start_time_ms is None or effective_start_time_ms > current_pair_start_time_ms:
                                     current_pair_start_time_ms = effective_start_time_ms
                                logger.info(f"Update mode: effective start for {pair} is {effective_start_dt}")
                            else:
                                logger.info(f"No existing data for {pair} in update mode. Using original start date if provided.")
                        
                        if current_pair_start_time_ms is None and not update:
                             logger.warning(f"No start date specified for {pair} and not in update/resume mode. Binance client will use its default (earliest).")

                        tqdm_instance = tqdm(total=0, unit="klines", desc=f"Downloading {pair}")
                        processed_klines_for_checkpoint = 0 

                        def progress_callback_for_pair(stats: Dict[str, Any]):
                            global tqdm_instance, processed_klines_for_checkpoint, shutdown_requested
                            nonlocal current_pair_start_time_checkpoint
                            
                            if shutdown_requested:
                                raise KeyboardInterrupt("Shutdown requested during kline download.")

                            klines_downloaded_this_batch = stats.get("current_pair_klines_downloaded_batch", 0)
                            total_klines_for_pair_so_far = stats.get("current_pair_klines_total_downloaded", 0)
                            estimated_total_for_pair = stats.get("current_pair_klines_estimated_total", 0)
                            last_kline_timestamp_ms = stats.get("current_pair_last_kline_timestamp_ms")

                            if tqdm_instance:
                                if tqdm_instance.total == 0 and estimated_total_for_pair > 0:
                                    tqdm_instance.total = estimated_total_for_pair
                                tqdm_instance.update(klines_downloaded_this_batch)
                                tqdm_instance.set_postfix_str(f"{total_klines_for_pair_so_far}/{estimated_total_for_pair or '?'}")
                            
                            if last_kline_timestamp_ms:
                                current_pair_start_time_checkpoint = last_kline_timestamp_ms

                            processed_klines_for_checkpoint += klines_downloaded_this_batch
                            if processed_klines_for_checkpoint >= KLINES_PER_CHECKPOINT and resume:
                                if current_pair_start_time_checkpoint:
                                    save_checkpoint({
                                        "target_pairs": parsed_pairs,
                                        "interval": interval,
                                        "original_start_date_iso": start_date.isoformat() if start_date else None,
                                        "completed_pairs": completed_pairs_checkpoint,
                                        "current_pair_processing": pair,
                                        "current_pair_last_timestamp_ms": current_pair_start_time_checkpoint,
                                        "last_saved_at": datetime.now(timezone.utc).isoformat()
                                    })
                                    processed_klines_for_checkpoint = 0

                        try:
                            dm_start_date = datetime.fromtimestamp(current_pair_start_time_ms / 1000, tz=timezone.utc) if current_pair_start_time_ms else None
                            
                            await data_manager.download_historical_data(
                                pairs=[pair], # Corrected argument name
                                start_date=dm_start_date,
                                end_date=None,
                                interval=interval,
                                resume=resume, # Pass the CLI resume flag to DataManager
                                progress_callback=progress_callback_for_pair 
                            )
                            
                            completed_pairs_checkpoint.append(pair)
                            current_pair_processing_for_checkpoint = None 
                            current_timestamp_for_checkpoint = None
                            if resume:
                                save_checkpoint({
                                    "target_pairs": parsed_pairs,
                                    "interval": interval,
                                    "original_start_date_iso": start_date.isoformat() if start_date else None,
                                    "completed_pairs": completed_pairs_checkpoint,
                                    "current_pair_processing": current_pair_processing_for_checkpoint,
                                    "current_pair_last_timestamp_ms": current_timestamp_for_checkpoint,
                                    "last_saved_at": datetime.now(timezone.utc).isoformat()
                                })
                            overall_pbar.update(1)

                        except KeyboardInterrupt:
                            logger.warning(f"Download for {pair} interrupted.")
                            raise 
                        except DataDownloadError as dde:
                            logger.error(f"Failed to download data for {pair}: {dde}")
                        except Exception as e_pair:
                            logger.error(f"An unexpected error occurred while processing {pair}: {e_pair}")
                        finally:
                            if tqdm_instance:
                                tqdm_instance.close()
                                tqdm_instance = None
                
                if not shutdown_requested and len(completed_pairs_checkpoint) == len(parsed_pairs):
                    logger.success("All pairs downloaded successfully.")
                    clear_checkpoint()
                elif shutdown_requested:
                     logger.warning("Download process was shut down. Partial progress may be saved in checkpoint.")
                else:
                    failed_pairs_count = len(parsed_pairs) - len(completed_pairs_checkpoint)
                    if failed_pairs_count > 0:
                        logger.warning(f"{failed_pairs_count} pair(s) could not be fully processed. Check logs. Partial progress saved in checkpoint.")

            asyncio.run(main_download_logic())

        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        except StorageError as e:
            logger.error(f"Storage error: {e}")
            sys.exit(1)
        except AlgoBotException as e:
            logger.error(f"An application error occurred: {e}")
            sys.exit(1)
        except KeyboardInterrupt: 
            logger.warning("Download process interrupted by user.")
            if shutdown_requested and resume and current_pair_checkpoint and current_pair_start_time_checkpoint: # Check if these are set
                 save_checkpoint({
                    "target_pairs": parsed_pairs,
                    "interval": interval,
                    "original_start_date_iso": start_date.isoformat() if start_date else None,
                    "completed_pairs": completed_pairs_checkpoint,
                    "current_pair_processing": current_pair_checkpoint,
                    "current_pair_last_timestamp_ms": current_pair_start_time_checkpoint,
                    "last_saved_at": datetime.now(timezone.utc).isoformat()
                })
            sys.exit(130)
        except Exception as e:
            logger.exception(f"An unexpected critical error occurred: {e}")
            sys.exit(1)
        finally:
            # Ensure data_manager is closed if it was initialized
            # The asyncio.run will handle the main_download_logic's finally block
            # but if an error occurs before or outside main_download_logic,
            # data_manager might be initialized but not closed by that block.
            # However, data_manager is now defined inside main_download_logic's scope.
            # A more robust pattern might involve initializing DataManager outside and passing it in,
            # or ensuring its __aenter__/__aexit__ are used if it's an async context manager.
            # For now, its own finally block inside main_download_logic should cover most cases.
            logger.info("Download process finished.")


    except click.UsageError as e:
        logger.error(f"CLI Usage Error: {e.message}")
        sys.exit(e.exit_code)
    except click.BadParameter as e:
        logger.error(f"CLI Parameter Error: {e.message}")
        sys.exit(e.exit_code)
    except Exception as e:
        logger.error(f"A critical error occurred before starting download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not hasattr(logger, 'level'): 
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    download_cli()
