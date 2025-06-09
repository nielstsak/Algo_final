# src/data/enriched_dataframe.py
import pandas as pd
import re
from typing import Set

from src.core.constants import Kline
from src.core.exceptions import DataError


class EnrichedDataFrame:
    """
    Encapsule un DataFrame enrichi contenant plusieurs vues de klines à différentes
    fréquences (par exemple, K_1m_close, K_5m_open) et fournit un accès standardisé.
    Cette classe est conçue pour être immuable afin d'empêcher la modification
    accidentelle des données partagées.
    """
    __slots__ = ('_df', '_frequencies')

    def __init__(self, df: pd.DataFrame):
        """
        Initialise l'EnrichedDataFrame.

        Args:
            df: Le DataFrame d'entrée. Doit avoir un DatetimeIndex.

        Raises:
            DataError: Si l'entrée n'est pas un DataFrame avec un DatetimeIndex.
        """
        if not isinstance(df, pd.DataFrame):
            raise DataError("Input must be a pandas DataFrame.")

        # Vérifier et définir l'index si nécessaire
        if isinstance(df.index, pd.DatetimeIndex):
            df_indexed = df
        elif 'timestamp' in df.columns:
            df_indexed = df.set_index('timestamp')
            if not isinstance(df_indexed.index, pd.DatetimeIndex):
                raise DataError("The 'timestamp' column could not be converted to a DatetimeIndex.")
        else:
            raise DataError("Input DataFrame must have a DatetimeIndex or a 'timestamp' column.")

        # Utilise un attribut privé, immuable par convention.
        object.__setattr__(self, '_df', df_indexed.copy())
        object.__setattr__(self, '_frequencies', self._detect_frequencies())

    def __setattr__(self, name, value):
        """Empêche la modification des attributs de l'instance après l'initialisation."""
        raise AttributeError("EnrichedDataFrame is immutable")

    @property
    def df(self) -> pd.DataFrame:
        """Retourne une copie du DataFrame interne pour garantir l'immuabilité."""
        return self._df.copy()

    @property
    def frequencies(self) -> Set[str]:
        """Retourne l'ensemble des fréquences disponibles."""
        return self._frequencies

    def _detect_frequencies(self) -> Set[str]:
        """
        Détecte les fréquences disponibles dans le DataFrame en analysant
        les noms de colonnes avec le motif 'K_<freq>_<ohlcv>'.
        """
        # Motif pour correspondre au préfixe et à la partie fréquence du nom de colonne.
        freq_pattern = re.compile(f"^{Kline.ROLLING_KLINE_PREFIX}_([a-zA-Z0-9]+)_(?:open|high|low|close|volume)")
        found_frequencies = set()
        for col in self._df.columns:
            if not isinstance(col, str): continue # Ignore les colonnes non-string
            match = freq_pattern.match(col)
            if match:
                found_frequencies.add(match.group(1))
        return found_frequencies

    def get_view(self, frequency: str) -> pd.DataFrame:
        """
        Extrait une vue pour une fréquence donnée avec des noms de colonnes OHLCV standard.
        Cette méthode crée un nouveau DataFrame contenant la vue ; elle ne retourne pas
        une tranche de l'original pour éviter les effets de bord.

        Args:
            frequency: La chaîne de fréquence (par ex., '1h', '5m').

        Returns:
            Un nouveau DataFrame pandas avec les colonnes 'open', 'high', 'low', 'close', 'volume'.

        Raises:
            ValueError: Si la fréquence demandée ou ses colonnes requises ne sont pas disponibles.
        """
        if frequency not in self.frequencies:
            raise ValueError(f"Frequency '{frequency}' not available. Found: {sorted(list(self.frequencies))}")

        # Utilise la constante helper pour créer le préfixe, par ex., 'K_5m_'
        prefix = f"{Kline.ROLLING_KLINE_PREFIX}_{frequency}_"
        
        # Mappage des noms de colonnes standardisés vers leurs équivalents préfixés
        required_cols_map = {
            'open': f"{prefix}{Kline.OHLCV_OPEN}",
            'high': f"{prefix}{Kline.OHLCV_HIGH}",
            'low': f"{prefix}{Kline.OHLCV_LOW}",
            'close': f"{prefix}{Kline.OHLCV_CLOSE}",
            'volume': f"{prefix}{Kline.OHLCV_VOLUME}"
        }

        # Vérifier les colonnes manquantes
        missing_cols = [prefixed for prefixed in required_cols_map.values() if prefixed not in self._df.columns]
        if missing_cols:
            raise ValueError(f"Columns for frequency '{frequency}' view are missing from EnrichedDataFrame: {missing_cols}")

        # Construire le DataFrame de la vue efficacement
        view_df = pd.DataFrame({
            std_name: self._df[prefixed_name]
            for std_name, prefixed_name in required_cols_map.items()
        }, index=self._df.index)

        return view_df
