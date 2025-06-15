import pandas as pd
import pandas_ta as ta
from loguru import logger
import numpy as np

from src.core.exceptions import IndicatorCalculationError

# ==============================================================================
# Module fonctionnel pour le calcul des indicateurs techniques.
# ==============================================================================

def calculate_bollinger_bands(close_series: pd.Series, period: int = 20, std_dev: float = 2.0, **kwargs) -> pd.DataFrame:
    """ Calcule les Bandes de Bollinger. """
    try:
        bbands_df = ta.bbands(close_series, length=period, std=std_dev, **kwargs)
        if bbands_df is None or bbands_df.empty:
            logger.warning("Le calcul des Bandes de Bollinger n'a retourné aucune donnée.")
            return pd.DataFrame(columns=['BBL', 'BBM', 'BBU'], index=close_series.index)
        rename_map = {
            f'BBL_{period}_{float(std_dev)}': 'BBL',
            f'BBM_{period}_{float(std_dev)}': 'BBM',
            f'BBU_{period}_{float(std_dev)}': 'BBU',
        }
        bbands_df.rename(columns=rename_map, inplace=True)
        return bbands_df[['BBL', 'BBM', 'BBU']]
    except Exception as e:
        logger.error(f"Erreur technique lors du calcul des Bandes de Bollinger : {e}")
        raise IndicatorCalculationError(f"Échec du calcul des Bandes de Bollinger: {e}") from e

def calculate_rsi(close_series: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """ Calcule le RSI. """
    try:
        rsi_series = ta.rsi(close_series, length=period, **kwargs)
        if rsi_series is None:
             logger.warning("Le calcul du RSI n'a retourné aucune donnée.")
             return pd.Series(dtype=float, index=close_series.index, name="RSI")
        rsi_series.name = "RSI"
        return rsi_series
    except Exception as e:
        logger.error(f"Erreur technique lors du calcul du RSI : {e}")
        raise IndicatorCalculationError(f"Échec du calcul du RSI: {e}") from e

def calculate_sma(series: pd.Series, period: int = 20, **kwargs) -> pd.Series:
    """ Calcule une Moyenne Mobile Simple (SMA). """
    try:
        sma_series = ta.sma(series, length=period, **kwargs)
        if sma_series is None:
             logger.warning(f"Le calcul de la SMA (période={period}) n'a retourné aucune donnée.")
             return pd.Series(dtype=float, index=series.index)
        return sma_series
    except Exception as e:
        logger.error(f"Erreur technique lors du calcul de la SMA : {e}")
        raise IndicatorCalculationError(f"Échec du calcul de la SMA: {e}") from e

def calculate_psar(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, **kwargs) -> pd.DataFrame:
    """ Calcule le Parabolic SAR. """
    try:
        psar_df = ta.psar(high_series, low_series, close_series, **kwargs)
        if psar_df is None or psar_df.empty:
            logger.warning("Le calcul du PSAR n'a retourné aucune donnée.")
            return pd.DataFrame(index=close_series.index)
        # Renomme les colonnes pour la simplicité
        psar_df.rename(columns=lambda x: x.lower().replace(f'_0.02_0.2',''), inplace=True)
        return psar_df
    except Exception as e:
        logger.error(f"Erreur technique lors du calcul du PSAR : {e}")
        raise IndicatorCalculationError(f"Échec du calcul du PSAR: {e}") from e

def calculate_atr(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int = 14, **kwargs) -> pd.Series:
    """ Calcule l'Average True Range (ATR). """
    try:
        atr_series = ta.atr(high_series, low_series, close_series, length=period, **kwargs)
        if atr_series is None:
            logger.warning("Le calcul de l'ATR n'a retourné aucune donnée.")
            return pd.Series(dtype=float, index=close_series.index, name="atr")
        atr_series.name = "atr"
        return atr_series
    except Exception as e:
        logger.error(f"Erreur technique lors du calcul de l'ATR : {e}")
        raise IndicatorCalculationError(f"Échec du calcul de l'ATR: {e}") from e

def calculate_otoco_pattern(df: pd.DataFrame, body_ratio_thld: float, wick_ratio_thld: float) -> pd.DataFrame:
    """
    Calcule le pattern OTOCO qui identifie des bougies de forte conviction.

    Args:
        df (pd.DataFrame): DataFrame avec les colonnes 'open', 'high', 'low', 'close'.
        body_ratio_thld (float): Seuil du ratio corps/range. Un ratio élevé
                                 indique un grand corps (forte conviction).
        wick_ratio_thld (float): Seuil du ratio mèche/range. Un ratio faible
                                 indique une petite mèche (clôture forte).

    Returns:
        pd.DataFrame: DataFrame avec les colonnes booléennes 'otoco_long' et 'otoco_short'.
    """
    signals = pd.DataFrame(index=df.index)

    # Pré-calcul des composantes de la bougie
    candle_range = df['high'] - df['low']
    real_body = (df['close'] - df['open']).abs()
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    # Éviter la division par zéro pour les bougies sans range (doji)
    candle_range_no_zero = candle_range.replace(0, np.nan)

    # Condition 1: Le corps doit être grand par rapport au range total.
    body_ratio_ok = (real_body / candle_range_no_zero) >= body_ratio_thld

    # Condition 2.1 (Long): Bougie haussière ET mèche supérieure petite.
    is_bullish = df['close'] > df['open']
    upper_wick_small = (upper_wick / candle_range_no_zero) <= wick_ratio_thld
    signals['otoco_long'] = is_bullish & body_ratio_ok & upper_wick_small

    # Condition 2.2 (Short): Bougie baissière ET mèche inférieure petite.
    is_bearish = df['close'] < df['open']
    lower_wick_small = (lower_wick / candle_range_no_zero) <= wick_ratio_thld
    signals['otoco_short'] = is_bearish & body_ratio_ok & lower_wick_small

    # Remplacer les NaN potentiels (dus à la division par zéro) par False.
    signals.fillna(False, inplace=True)
    
    logger.info("Calcul du pattern OTOCO terminé.")

    return signals
