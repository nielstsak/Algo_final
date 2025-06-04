import inspect
from pathlib import Path
from loguru import logger
import sys

# Configure logger minimally for this test
logger.remove()
logger.add(sys.stderr, level="DEBUG")

try:
    logger.debug("Attempting to import ParquetStorage...")
    # Adjust the import path if your PROJECT_ROOT isn't directly on PYTHONPATH
    # For now, assuming direct relative import works from project root for src.
    from src.data.storage.parquet_storage import ParquetStorage
    from src.core.config import settings # To get settings.data.storage_path

    logger.debug(f"Successfully imported ParquetStorage.")
    logger.debug(f"ParquetStorage type: {type(ParquetStorage)}")
    logger.debug(f"ParquetStorage module: {ParquetStorage.__module__}")
    logger.debug(f"File path from inspect: {inspect.getfile(ParquetStorage)}")
    logger.debug(f"Signature of __init__: {inspect.signature(ParquetStorage.__init__)}")

    logger.debug("Attempting to instantiate ParquetStorage...")
    # Use a dummy Path or the actual settings path for instantiation
    dummy_path = settings.data.storage_path # Or Path("./dummy_storage")
    storage_instance = ParquetStorage(dummy_path)
    logger.success(f"Successfully instantiated ParquetStorage with path: {dummy_path}")
    logger.debug(f"Instance type: {type(storage_instance)}")

except ImportError as e:
    logger.error(f"ImportError: {e}")
    logger.error("Ensure your PYTHONPATH is set up correctly or run this script from the project root.")
    logger.error(f"Current sys.path: {sys.path}")
except Exception as e:
    logger.exception("An error occurred during the test:")