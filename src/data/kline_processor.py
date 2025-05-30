# src/data/kline_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from loguru import logger

from src.core.constants import Kline
from src.core.exceptions import DataError, KlineValidationError


class KlineProcessor:
    """
    Processeur pour la manipulation et le resampling des klines.
    Gère la génération de klines multi-fréquences (standard et glissantes).
    Implémente DAT-016, DAT-017, DAT-018, DAT-020.
    L'optimisation avec NumPy (DAT-019) est implicite via l'utilisation de Pandas.
    """

    def __init__(self):
        """Initialise le processeur de klines."""
        # Mapping des intervalles Binance vers les fréquences pandas
        self.interval_mapping: Dict[str, str] = {
            Kline.INTERVAL_1MINUTE: '1T',
            Kline.INTERVAL_3MINUTE: '3T',
            Kline.INTERVAL_5MINUTE: '5T',
            Kline.INTERVAL_15MINUTE: '15T',
            Kline.INTERVAL_30MINUTE: '30T',
            Kline.INTERVAL_1HOUR: '1H',
            Kline.INTERVAL_2HOUR: '2H',
            Kline.INTERVAL_4HOUR: '4H',
            Kline.INTERVAL_6HOUR: '6H',
            Kline.INTERVAL_8HOUR: '8H',
            Kline.INTERVAL_12HOUR: '12H',
            Kline.INTERVAL_1DAY: '1D',
            Kline.INTERVAL_3DAY: '3D',
            Kline.INTERVAL_1WEEK: '1W',
            Kline.INTERVAL_1MONTH: '1MS'  # 'MS' pour début de mois, plus standard pour resampling mensuel
        }

        # Règles d'agrégation standard pour OHLCV et autres champs pertinents
        # S'assurer que les noms de colonnes correspondent à ceux utilisés dans DataManager
        self.aggregation_rules: Dict[str, Union[str, Callable]] = {
            Kline.OHLCV_OPEN: 'first',
            Kline.OHLCV_HIGH: 'max',
            Kline.OHLCV_LOW: 'min',
            Kline.OHLCV_CLOSE: 'last',
            Kline.OHLCV_VOLUME: 'sum', # Correspond à 'base_asset_volume' dans DataManager
            Kline.OHLCV_QUOTE_ASSET_VOLUME: 'sum',
            Kline.OHLCV_NUMBER_OF_TRADES: 'sum',
            # Les colonnes taker_buy_... sont aussi sommées
            Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: 'sum',
            Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: 'sum',
            # 'pair' et 'is_kline_closed' sont généralement gérés séparément ou pris du premier/dernier
            'pair': 'first', # Si la colonne 'pair' est présente et qu'on resample un df multi-paires
            'is_kline_closed': 'last' # Pour les klines resamplées, elles sont considérées fermées
        }

        logger.debug("KlineProcessor initialized")

    def _validate_input_dataframe(self, df: pd.DataFrame, expected_freq_minutes: int = 1) -> None:
        """Valide que le DataFrame d'entrée est correct pour le resampling."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise KlineValidationError("Input DataFrame must have a DatetimeIndex.")
        if df.empty:
            raise KlineValidationError("Input DataFrame is empty.")

        # Vérifier si les colonnes nécessaires pour l'agrégation existent
        required_cols_for_agg = [
            Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME
        ]
        missing_cols = [col for col in required_cols_for_agg if col not in df.columns]
        if missing_cols:
            raise KlineValidationError(f"Input DataFrame is missing required columns for aggregation: {missing_cols}")

        # Vérifier la fréquence de l'index si possible (approximatif)
        if len(df.index) > 1:
            median_diff = df.index.to_series().diff().median()
            expected_timedelta = pd.Timedelta(minutes=expected_freq_minutes)
            # Permettre une petite tolérance pour les irrégularités mineures
            if median_diff > expected_timedelta * 1.1 or median_diff < expected_timedelta * 0.9 :
                logger.warning(
                    f"Input DataFrame median frequency ({median_diff}) "
                    f"differs from expected ({expected_timedelta}). Ensure it's sorted 1-minute data."
                )

    def _validate_ohlcv(self, df: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """
        Valide les données OHLCV d'un DataFrame.
        Logue les erreurs et supprime les lignes invalides.
        DAT-020
        """
        if df.empty:
            return df

        initial_len = len(df)
        original_df = df.copy() # Copie pour comparaison si des lignes sont supprimées

        # Renommer temporairement si les colonnes sont préfixées ou différentes
        col_map = {
            # Potentiels alias -> Nom standard utilisé dans les constantes Kline
            'open_price': Kline.OHLCV_OPEN,
            'high_price': Kline.OHLCV_HIGH,
            'low_price': Kline.OHLCV_LOW,
            'close_price': Kline.OHLCV_CLOSE,
            'base_asset_volume': Kline.OHLCV_VOLUME,
        }
        # Appliquer le renommage uniquement si nécessaire et si les colonnes existent
        rename_dict = {k: v for k, v in col_map.items() if k in df.columns and v not in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)

        # Conditions d'invalidité
        conditions = (
            (df[Kline.OHLCV_HIGH] < df[Kline.OHLCV_LOW]) |
            (df[Kline.OHLCV_HIGH] < df[Kline.OHLCV_OPEN]) |
            (df[Kline.OHLCV_HIGH] < df[Kline.OHLCV_CLOSE]) |
            (df[Kline.OHLCV_LOW] > df[Kline.OHLCV_OPEN]) |
            (df[Kline.OHLCV_LOW] > df[Kline.OHLCV_CLOSE]) |
            (df[Kline.OHLCV_VOLUME] < 0) |
            df[[Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE, Kline.OHLCV_VOLUME]].isnull().any(axis=1)
        )

        invalid_rows = df[conditions]
        if not invalid_rows.empty:
            logger.warning(f"{context} Found {len(invalid_rows)} invalid OHLCV rows. Examples:\n{invalid_rows.head()}")
            df = df[~conditions] # Conserver uniquement les lignes valides

        if len(df) < initial_len:
             logger.warning(f"{context} Removed {initial_len - len(df)} invalid rows during OHLCV validation.")

        # Revenir aux noms de colonnes d'origine si un renommage a eu lieu
        # Ceci est important si le DataFrame retourné doit conserver le schéma d'origine.
        # Pour cette fonction, nous retournons avec les noms standard de Kline.OHLCV_*.
        # Si le DataFrame d'origine doit être préservé tel quel, il faudrait une logique de réversion plus complexe.
        # Pour l'instant, on s'attend à ce que le df en sortie ait les colonnes Kline.OHLCV_*.

        return df

    def _get_pandas_freq(self, target_freq_binance: str) -> str:
        """Convertit un intervalle de type Binance en fréquence Pandas."""
        pandas_freq = self.interval_mapping.get(target_freq_binance)
        if not pandas_freq:
            raise ValueError(f"Unsupported target frequency: {target_freq_binance}. "
                             f"Supported: {list(self.interval_mapping.keys())}")
        return pandas_freq

    def _generate_standard_klines(self, df: pd.DataFrame, pandas_freq: str) -> pd.DataFrame:
        """
        Klines classiques avec pandas resample.
        DAT-017
        """
        logger.debug(f"Generating standard klines for frequency: {pandas_freq}")

        # S'assurer que les colonnes à agréger existent, sinon les exclure des règles
        agg_rules_to_apply = {
            col: rule for col, rule in self.aggregation_rules.items() if col in df.columns
        }
        if not agg_rules_to_apply:
            raise DataError("No columns available for aggregation in standard kline generation.")

        # 'label' et 'closed' déterminent de quel côté l'intervalle est fermé et où le timestamp est placé.
        # 'left' pour label et closed signifie que le timestamp de la barre resamplée est celui du début
        # de l'intervalle, et l'intervalle inclut le côté gauche.
        resampled_df = df.resample(pandas_freq, label='left', closed='left').agg(agg_rules_to_apply)
        
        # Remplir 'pair' si elle a été perdue et qu'elle était unique
        if 'pair' not in resampled_df.columns and 'pair' in df.columns and df['pair'].nunique() == 1:
            resampled_df['pair'] = df['pair'].iloc[0]
        
        # Les klines resamplées sont par définition fermées
        resampled_df['is_kline_closed'] = True
        
        return resampled_df.dropna(subset=[Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE]) # Enlever les lignes où open/close est NaN

    def _generate_rolling_klines(self, df: pd.DataFrame, pandas_freq: str, window_size_minutes: int) -> pd.DataFrame:
        """
        Klines glissantes pour analyse plus granulaire.
        Chaque kline de 1m devient le point de fin d'une kline de 'window_size_minutes'.
        DAT-018
        """
        logger.debug(f"Generating rolling klines for window: {window_size_minutes} minutes (target freq: {pandas_freq})")

        if window_size_minutes <= 0:
            raise ValueError("Window size for rolling klines must be positive.")

        # S'assurer que les colonnes à agréger existent
        agg_rules_to_apply = {
            col: rule for col, rule in self.aggregation_rules.items() if col in df.columns
        }
        if not agg_rules_to_apply:
            raise DataError("No columns available for aggregation in rolling kline generation.")

        # Le rolling s'applique sur un nombre de périodes.
        # Si df est en 1min, window=N signifie N minutes.
        rolling_window = df.rolling(window=f"{window_size_minutes}T") # Utilise la chaîne de temps pour la fenêtre
        
        # Appliquer les agrégations.
        # Note: l'agrégation 'first' pour 'open' dans une fenêtre glissante
        # signifie le 'open' de la première bougie de la fenêtre.
        # 'last' pour 'close' signifie le 'close' de la dernière bougie de la fenêtre.
        # 'max' pour 'high' signifie le 'high' maximum sur toutes les bougies de la fenêtre.
        # 'min' pour 'low' signifie le 'low' minimum sur toutes les bougies de la fenêtre.
        
        # Correctement appliquer les agrégations pour rolling
        # open = df[Kline.OHLCV_OPEN].rolling(window=window_size_minutes).apply(lambda x: x[0], raw=True) # 'first'
        # high = df[Kline.OHLCV_HIGH].rolling(window=window_size_minutes).max()
        # low = df[Kline.OHLCV_LOW].rolling(window=window_size_minutes).min()
        # close = df[Kline.OHLCV_CLOSE].rolling(window=window_size_minutes).apply(lambda x: x[-1], raw=True) # 'last'
        # volume = df[Kline.OHLCV_VOLUME].rolling(window=window_size_minutes).sum()
        # # ... etc pour les autres colonnes

        # Une approche plus générique avec rolling().agg() si possible,
        # mais 'first' et 'last' ne sont pas des fonctions directes dans rolling.agg comme dans resample.
        # On doit les implémenter ou utiliser .apply()
        
        # Création d'un DataFrame résultat
        resampled_df = pd.DataFrame(index=df.index)

        if Kline.OHLCV_OPEN in agg_rules_to_apply:
            resampled_df[Kline.OHLCV_OPEN] = df[Kline.OHLCV_OPEN].rolling(window=window_size_minutes).apply(lambda x: x[0], raw=True)
        if Kline.OHLCV_HIGH in agg_rules_to_apply:
            resampled_df[Kline.OHLCV_HIGH] = df[Kline.OHLCV_HIGH].rolling(window=window_size_minutes).max()
        if Kline.OHLCV_LOW in agg_rules_to_apply:
            resampled_df[Kline.OHLCV_LOW] = df[Kline.OHLCV_LOW].rolling(window=window_size_minutes).min()
        if Kline.OHLCV_CLOSE in agg_rules_to_apply:
            resampled_df[Kline.OHLCV_CLOSE] = df[Kline.OHLCV_CLOSE].rolling(window=window_size_minutes).apply(lambda x: x[-1], raw=True)
        
        for col, rule in agg_rules_to_apply.items():
            if col not in [Kline.OHLCV_OPEN, Kline.OHLCV_HIGH, Kline.OHLCV_LOW, Kline.OHLCV_CLOSE]: # Déjà traités
                if rule == 'sum':
                    resampled_df[col] = df[col].rolling(window=window_size_minutes).sum()
                elif rule == 'first' and col in df.columns: # Ex: 'pair'
                     resampled_df[col] = df[col].rolling(window=window_size_minutes).apply(lambda x: x[0], raw=True)
                elif rule == 'last' and col in df.columns: # Ex: 'is_kline_closed'
                     resampled_df[col] = df[col].rolling(window=window_size_minutes).apply(lambda x: x[-1], raw=True)
                # D'autres règles comme 'mean', 'median', 'std', 'var' peuvent être ajoutées si nécessaire

        # Remplir 'pair' si elle a été perdue et qu'elle était unique
        if 'pair' not in resampled_df.columns and 'pair' in df.columns and df['pair'].nunique() == 1:
             resampled_df['pair'] = df['pair'].iloc[0] # Ou .rolling().apply(lambda x: x[0])

        resampled_df['is_kline_closed'] = True # Les klines roulantes sont basées sur des klines 1m fermées
        
        return resampled_df.dropna(subset=[Kline.OHLCV_OPEN, Kline.OHLCV_CLOSE])


    def resample_klines(
        self,
        df: pd.DataFrame,
        target_freq_binance: str,
        rolling: bool = False,
        window_size_minutes: Optional[int] = None,
        base_freq_minutes: int = 1
    ) -> pd.DataFrame:
        """
        Génère des klines de fréquence supérieure à partir de klines de base_freq_minutes (par défaut 1m).
        Permet de générer des klines standard (ex: 5m fixes) ou glissantes (ex: kline de 5m se terminant chaque minute).

        Args:
            df: DataFrame avec klines source (doit avoir un index DatetimeIndex et être trié).
                Les colonnes attendues sont définies dans Kline.OHLCV_*.
            target_freq_binance: Fréquence cible au format Binance ('3m', '5m', '1h', etc.).
            rolling: Si True, génère des klines glissantes.
                     Si False, génère des klines standard (non chevauchantes).
            window_size_minutes: Taille de la fenêtre en minutes pour les klines glissantes.
                                 Si None et rolling=True, déduit de target_freq_binance.
                                 Ex: si target_freq_binance='5m', window_size_minutes sera 5.
            base_freq_minutes: Fréquence en minutes des données d'entrée (par défaut 1 pour klines 1m).

        Returns:
            DataFrame avec klines resamplées.

        Raises:
            DataError: Si la fréquence cible n'est pas supportée ou si les données sont inadéquates.
            KlineValidationError: Si le DataFrame d'entrée est invalide.
            ValueError: Pour des paramètres invalides.
        """
        logger.info(
            f"Resampling klines to target_freq='{target_freq_binance}', "
            f"rolling={rolling}, window_size_minutes={window_size_minutes}"
        )
        self._validate_input_dataframe(df, expected_freq_minutes=base_freq_minutes)
        
        # Valider les données OHLCV d'entrée
        # Les colonnes doivent être nommées selon Kline.OHLCV_* pour _validate_ohlcv
        # Si DataManager fournit des colonnes comme 'open_price', il faut les mapper ou s'assurer de la cohérence.
        # Pour l'instant, on suppose que _validate_ohlcv gère les alias ou que df a déjà les bons noms.
        df_validated = self._validate_ohlcv(df, context="Input to resample_klines:")
        if df_validated.empty:
            logger.warning("Input DataFrame became empty after initial validation, returning empty DataFrame.")
            return pd.DataFrame()

        pandas_freq = self._get_pandas_freq(target_freq_binance)

        if rolling:
            if window_size_minutes is None:
                # Déduire window_size_minutes de target_freq_binance (ex: '5m' -> 5 minutes)
                try:
                    # Tente d'extraire le nombre de l'intervalle (ex: '5m' -> 5, '1h' -> 1*60)
                    if target_freq_binance.endswith('m'):
                        window_size_minutes = int(target_freq_binance[:-1])
                    elif target_freq_binance.endswith('h'):
                        window_size_minutes = int(target_freq_binance[:-1]) * 60
                    elif target_freq_binance.endswith('d'):
                        window_size_minutes = int(target_freq_binance[:-1]) * 60 * 24
                    else: # Fallback ou cas non gérés simplement
                        delta = pd.to_timedelta(pandas_freq)
                        window_size_minutes = int(delta.total_seconds() / 60)
                except ValueError as e:
                    raise ValueError(
                        f"Could not automatically determine window_size_minutes from target_freq_binance='{target_freq_binance}'. "
                        f"Please specify window_size_minutes. Original error: {e}"
                    )

            if window_size_minutes < base_freq_minutes or window_size_minutes % base_freq_minutes != 0:
                raise ValueError(
                    f"window_size_minutes ({window_size_minutes}) must be a multiple of base_freq_minutes ({base_freq_minutes})."
                )
            
            # Pour rolling, la fenêtre est exprimée en nombre de périodes de base_freq_minutes
            # Example: df is 1-min. window_size_minutes = 5 -> rolling window = 5 periods.
            num_base_periods_in_window = window_size_minutes // base_freq_minutes
            resampled_df = self._generate_rolling_klines(df_validated, pandas_freq, num_base_periods_in_window)
        else:
            # Pour resample standard, la fréquence doit être supérieure à la fréquence de base
            # Cette vérification est implicite car pandas_freq sera plus grand que base_freq_minutes
            # Ex: '1T' (base) vs '5T' (target)
            resampled_df = self._generate_standard_klines(df_validated, pandas_freq)

        # Valider les données OHLCV de sortie
        resampled_df_validated = self._validate_ohlcv(resampled_df, context=f"Output of resample_klines ({target_freq_binance}):")
        
        logger.success(
            f"Successfully resampled {len(df_validated)} input klines to {len(resampled_df_validated)} "
            f"{target_freq_binance} klines (rolling={rolling})."
        )
        return resampled_df_validated


if __name__ == '__main__':
    # Configuration de Loguru pour l'exemple (normalement fait dans main)
    logger.remove()
    logger.add(lambda msg: print(msg), level="DEBUG")

    # Création d'un exemple de DataFrame (klines 1 minute)
    # Colonnes comme définies dans Kline.OHLCV_*
    data = {
        Kline.OHLCV_OPEN: [10, 11, 10, 12, 11, 13, 14, 13, 15, 16],
        Kline.OHLCV_HIGH: [12, 12, 11, 13, 12, 14, 15, 14, 17, 18],
        Kline.OHLCV_LOW: [9, 10, 9, 11, 10, 12, 13, 12, 14, 15],
        Kline.OHLCV_CLOSE: [11, 10, 11, 11, 12, 13, 13, 14, 16, 17],
        Kline.OHLCV_VOLUME: [100, 110, 90, 120, 105, 115, 125, 95, 130, 135],
        Kline.OHLCV_QUOTE_ASSET_VOLUME: [1000, 1100, 900, 1200, 1050, 1150, 1250, 950, 1300, 1350],
        Kline.OHLCV_NUMBER_OF_TRADES: [10,11,9,12,10,11,12,9,13,13],
        Kline.OHLCV_TAKER_BUY_BASE_ASSET_VOLUME: [50,55,45,60,52,57,62,47,65,67],
        Kline.OHLCV_TAKER_BUY_QUOTE_ASSET_VOLUME: [500,550,450,600,520,570,620,470,650,670],
        'pair': ['BTCUSDC'] * 10,
        'is_kline_closed': [True] * 10
    }
    index = pd.to_datetime([f'2023-01-01 10:{i:02d}:00' for i in range(10)], utc=True)
    df_1m = pd.DataFrame(data, index=index)

    processor = KlineProcessor()

    logger.info("--- Standard Resampling (3 minutes) ---")
    df_3m_std = processor.resample_klines(df_1m.copy(), target_freq_binance='3m', rolling=False)
    print(df_3m_std)
    # Attendu: klines pour 10:00, 10:03, 10:06, 10:09

    logger.info("\n--- Rolling Resampling (3 minutes window) ---")
    df_3m_roll = processor.resample_klines(df_1m.copy(), target_freq_binance='3m', rolling=True)
    # ou explicitement: processor.resample_klines(df_1m.copy(), target_freq_binance='3m', rolling=True, window_size_minutes=3)
    print(df_3m_roll)
    # Attendu: klines pour chaque minute, représentant les 3 minutes précédentes (ou moins si au début)

    logger.info("\n--- Standard Resampling (invalid high<low data) ---")
    df_1m_invalid = df_1m.copy()
    df_1m_invalid.loc[df_1m_invalid.index[1], Kline.OHLCV_HIGH] = 8 # high < low, open, close
    df_3m_std_invalid_handled = processor.resample_klines(df_1m_invalid, target_freq_binance='3m', rolling=False)
    print(df_3m_std_invalid_handled)

    logger.info("\n--- Test avec un DataFrame vide ---")
    try:
        processor.resample_klines(pd.DataFrame(columns=df_1m.columns), target_freq_binance='5m')
    except KlineValidationError as e:
        logger.error(f"Error with empty DataFrame: {e}")

    logger.info("\n--- Test avec colonnes manquantes ---")
    try:
        processor.resample_klines(df_1m[['open', 'close']].copy(), target_freq_binance='5m')
    except KlineValidationError as e:
        logger.error(f"Error with missing columns: {e}")