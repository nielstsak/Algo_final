import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union

from src.strategies.base import BaseStrategy
from src.strategies.params import (
    BbandsVolumeRsiStrategyFixedParams,
    BbandsVolumeRsiStrategyOptimizableParams,
)
from src.strategies import technical_indicators as ti
from src.data.enriched_dataframe import EnrichedDataFrame

logger = logging.getLogger(__name__)


class BbandsVolumeRsiStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur la cassure (breakout) des Bandes de Bollinger,
    confirmée par le volume et le RSI. La gestion des risques (Stop-Loss et
    Take-Profit) est assurée dynamiquement par l'indicateur Average True Range (ATR).

    Cette stratégie identifie des points d'entrée potentiels lorsque le prix clôture
    au-dessus (pour un achat) ou en dessous (pour une vente) des Bandes de Bollinger,
    avec un volume supérieur à sa moyenne mobile et un RSI confirmant la dynamique.

    **Logique des signaux :**
    - **Achat (Long) :**
        1. Clôture du prix > Bande de Bollinger supérieure.
        2. Volume > Moyenne mobile du volume.
        3. RSI > Seuil de surachat (ex: 60).
    - **Vente (Short) :**
        1. Clôture du prix < Bande de Bollinger inférieure.
        2. Volume > Moyenne mobile du volume.
        3. RSI < Seuil de survente (ex: 40).

    Le Stop-Loss et le Take-Profit sont calculés en utilisant un multiple de l'ATR
    au moment de l'entrée pour s'adapter à la volatilité du marché.
    """

    name: str = "BbandsVolumeRsiStrategy"
    version: str = "1.1.0"
    description: str = (
        "Bollinger Bands, Volume & RSI Breakout Strategy with ATR-based SL/TP."
    )

    fixed_params_model = BbandsVolumeRsiStrategyFixedParams
    optimizable_params_model = BbandsVolumeRsiStrategyOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialise la stratégie avec des paramètres spécifiques."""
        super().__init__(params=params, **kwargs)

    def calculate_indicators(
        self, data: Union[pd.DataFrame, EnrichedDataFrame]
    ) -> pd.DataFrame:
        """
        Calcule tous les indicateurs techniques nécessaires à la stratégie.
        Cette méthode respecte la signature de la méthode abstraite de BaseStrategy.

        Args:
            data: DataFrame ou EnrichedDataFrame contenant les données de marché (OHLCV).

        Returns:
            Un DataFrame contenant les données originales et les indicateurs calculés.
        """
        df = self._prepare_indicator_data(data)
        if df.empty:
            logger.warning(
                "Le DataFrame de données est vide. Aucun indicateur ne sera calculé."
            )
            return pd.DataFrame()

        # Calcul des Bandes de Bollinger
        bbands_df = ti.calculate_bollinger_bands(
            close_series=df["close"],
            period=self.get_param("bbands_period"),
            std_dev=self.get_param("bbands_std_dev"),
        )

        # Calcul de la moyenne mobile du volume
        volume_ma = ti.calculate_sma(
            series=df["volume"], period=self.get_param("volume_ma_period")
        )
        volume_ma.name = "volume_ma"

        # Calcul du RSI
        rsi = ti.calculate_rsi(
            close_series=df["close"], period=self.get_param("rsi_period")
        )
        rsi.name = "rsi"

        # Calcul de l'ATR pour la gestion des risques
        atr = ti.calculate_atr(
            high_series=df["high"],
            low_series=df["low"],
            close_series=df["close"],
            period=self.get_param("atr_period"),
        )
        atr.name = "atr"

        # Fusion de tous les indicateurs dans un seul DataFrame
        indicators_df = pd.concat([df, bbands_df, volume_ma, rsi, atr], axis=1)
        logger.info(
            "Indicateurs calculés avec succès pour la stratégie de breakout."
        )
        return indicators_df

    def generate_signals(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les signaux de trading (entrées, sorties, SL/TP) basés sur la
        logique de la stratégie et les indicateurs fournis.

        Args:
            indicators_df: DataFrame contenant les données de marché et les indicateurs.

        Returns:
            Un DataFrame contenant les colonnes de signaux requises.
        """
        if indicators_df.empty:
            logger.warning(
                "Le DataFrame d'indicateurs est vide. Impossible de générer des signaux."
            )
            return pd.DataFrame()

        df = indicators_df

        # --- Conditions pour les signaux d'achat (Long) ---
        long_breakout_cond = df["close"] > df["BBU"]
        long_volume_cond = df["volume"] > df["volume_ma"]
        long_rsi_cond = df["rsi"] > self.get_param("rsi_buy_breakout_threshold")

        all_long_conditions = (
            long_breakout_cond & long_volume_cond & long_rsi_cond
        )

        # On ne prend que la première occurrence de la condition
        entry_long = all_long_conditions & ~all_long_conditions.shift(
            1, fill_value=False
        )

        # --- Conditions pour les signaux de vente (Short) ---
        short_breakout_cond = df["close"] < df["BBL"]
        short_volume_cond = df["volume"] > df["volume_ma"]
        short_rsi_cond = df["rsi"] < self.get_param(
            "rsi_sell_breakout_threshold"
        )

        all_short_conditions = (
            short_breakout_cond & short_volume_cond & short_rsi_cond
        )
        
        # On ne prend que la première occurrence de la condition
        entry_short = all_short_conditions & ~all_short_conditions.shift(
            1, fill_value=False
        )

        # --- Construction du DataFrame de signaux ---
        signals = pd.DataFrame(index=df.index)
        signals["entry_long"] = entry_long
        signals["entry_short"] = entry_short
        
        # Les signaux de sortie ne sont pas définis par cette stratégie,
        # le backtest se basera sur SL/TP.
        signals["exit_long"] = False
        signals["exit_short"] = False

        # --- Calcul des niveaux de Stop-Loss et Take-Profit ---
        sl_atr_mult = self.get_param("sl_atr_mult")
        tp_atr_mult = self.get_param("tp_atr_mult")

        # SL/TP pour les positions longues
        signals["sl_long"] = np.where(
            entry_long, df["close"] - (df["atr"] * sl_atr_mult), np.nan
        )
        signals["tp_long"] = np.where(
            entry_long, df["close"] + (df["atr"] * tp_atr_mult), np.nan
        )

        # SL/TP pour les positions courtes
        signals["sl_short"] = np.where(
            entry_short, df["close"] + (df["atr"] * sl_atr_mult), np.nan
        )
        signals["tp_short"] = np.where(
            entry_short, df["close"] - (df["atr"] * tp_atr_mult), np.nan
        )
        
        # Fusionner les colonnes SL/TP pour le backtesting engine
        signals["sl"] = signals["sl_long"].fillna(signals["sl_short"])
        signals["tp"] = signals["tp_long"].fillna(signals["tp_short"])

        signals.drop(
            columns=["sl_long", "tp_long", "sl_short", "tp_short"], inplace=True
        )

        logger.info(
            f"Signaux générés : {entry_long.sum()} signaux d'achat, {entry_short.sum()} signaux de vente."
        )
        return signals
