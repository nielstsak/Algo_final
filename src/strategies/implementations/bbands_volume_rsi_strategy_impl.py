import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.strategies.base_strategy import BaseStrategy
from src.strategies.params import BbandsVolumeRsiStrategyFixedParams, BbandsVolumeRsiStrategyOptimizableParams
from src.strategies import technical_indicators as ti

logger = logging.getLogger(__name__)

class BbandsVolumeRsiStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur la cassure (breakout) des Bandes de Bollinger,
    confirmée par le volume et le RSI. La gestion des risques (Stop-Loss et
    Take-Profit) est assurée dynamiquement par l'indicateur Average True Range (ATR).
    """
    name: str = "BbandsVolumeRsiStrategy"
    version: str = "1.0.0"
    description: str = "Bollinger Bands, Volume & RSI Breakout Strategy with ATR-based SL/TP."

    fixed_params_model = BbandsVolumeRsiStrategyFixedParams
    optimizable_params_model = BbandsVolumeRsiStrategyOptimizableParams

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialise la stratégie avec les paramètres fournis et validés.

        Args:
            params: Dictionnaire optionnel de paramètres pour surcharger les valeurs par défaut.
            **kwargs: Arguments supplémentaires passés à BaseStrategy (ex: pair_symbol).
        """
        super().__init__(params=params, **kwargs)
        # self.params est maintenant un dictionnaire contenant les paramètres validés et fusionnés.
        # La validation est gérée par BaseStrategy en utilisant fixed_params_model et optimizable_params_model.
        self.indicators_df = pd.DataFrame()

    def _calculate_indicators(self, data: pd.DataFrame) -> None:
        """
        Calcule tous les indicateurs techniques nécessaires à la stratégie
        et les stocke dans l'attribut `self.indicators_df`.

        Args:
            data: DataFrame contenant les données de marché (OHLCV).
        """
        if data.empty:
            logger.warning("Le DataFrame de données est vide. Aucun indicateur ne sera calculé.")
            self.indicators_df = pd.DataFrame()
            return

        # Calcul des Bandes de Bollinger
        bbands_df = ti.calculate_bbands(
            close=data['close'],
            period=self.params.bbands_period,
            std_dev=self.params.bbands_std_dev
        )

        # Calcul de la moyenne mobile du volume
        volume_ma = ti.calculate_sma(
            data=data['volume'],
            period=self.params.volume_ma_period
        )
        volume_ma.name = 'volume_ma'

        # Calcul du RSI
        rsi = ti.calculate_rsi(
            close=data['close'],
            period=self.params.rsi_period
        )
        rsi.name = 'rsi'
        
        # Calcul de l'ATR pour la gestion des risques
        atr = ti.calculate_atr(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            period=self.params.atr_period
        )
        atr.name = 'atr'

        # Fusion de tous les indicateurs dans un seul DataFrame
        self.indicators_df = pd.concat([data, bbands_df, volume_ma, rsi, atr], axis=1)
        logger.info("Indicateurs calculés avec succès pour la stratégie de breakout.")

    def generate_signals(self, data: pd.DataFrame) -> None:
        """
        Génère les signaux de trading (entrées, sorties, SL/TP) basés sur la
        logique de la stratégie et les stocke dans `self._signals`.

        Args:
            data: DataFrame contenant les données de marché (OHLCV).
        """
        self._calculate_indicators(data)

        if self.indicators_df.empty:
            logger.warning("Le DataFrame d'indicateurs est vide. Impossible de générer des signaux.")
            self._signals = pd.DataFrame()
            return

        df = self.indicators_df
        
        # --- Conditions pour les signaux d'achat (Long) ---
        long_breakout_cond = df['close'] > df['bb_upper']
        long_volume_cond = df['volume'] > df['volume_ma']
        long_rsi_cond = df['rsi'] > self.params.rsi_buy_breakout_threshold
        
        all_long_conditions = long_breakout_cond & long_volume_cond & long_rsi_cond
        
        # Le signal se déclenche uniquement sur la première bougie qui remplit les conditions
        entry_long = all_long_conditions & ~all_long_conditions.shift(1).fillna(False)

        # --- Conditions pour les signaux de vente (Short) ---
        short_breakout_cond = df['close'] < df['bb_lower']
        short_volume_cond = df['volume'] > df['volume_ma']
        short_rsi_cond = df['rsi'] < self.params.rsi_sell_breakout_threshold
        
        all_short_conditions = short_breakout_cond & short_volume_cond & short_rsi_cond

        # Le signal se déclenche uniquement sur la première bougie qui remplit les conditions
        entry_short = all_short_conditions & ~all_short_conditions.shift(1).fillna(False)
        
        # --- Construction du DataFrame de signaux ---
        signals = pd.DataFrame(index=df.index)
        signals['entry_long'] = entry_long
        signals['entry_short'] = entry_short
        
        # Les sorties sont gérées par SL/TP, donc ces signaux sont toujours False
        signals['exit_long'] = False
        signals['exit_short'] = False

        # --- Calcul des niveaux de Stop-Loss et Take-Profit ---
        # SL/TP ne sont calculés que sur les bougies d'entrée
        
        # Pour les positions longues
        signals['sl_long'] = np.where(
            entry_long,
            df['close'] - (df['atr'] * self.params.sl_atr_mult),
            np.nan
        )
        signals['tp_long'] = np.where(
            entry_long,
            df['close'] + (df['atr'] * self.params.tp_atr_mult),
            np.nan
        )
        
        # Pour les positions courtes
        signals['sl_short'] = np.where(
            entry_short,
            df['close'] + (df['atr'] * self.params.sl_atr_mult),
            np.nan
        )
        signals['tp_short'] = np.where(
            entry_short,
            df['close'] - (df['atr'] * self.params.tp_atr_mult),
            np.nan
        )
        
        # Fusion des SL/TP dans des colonnes uniques pour le moteur de backtesting
        signals['sl'] = signals['sl_long'].fillna(signals['sl_short'])
        signals['tp'] = signals['tp_long'].fillna(signals['tp_short'])

        # Nettoyage des colonnes intermédiaires
        signals.drop(columns=['sl_long', 'tp_long', 'sl_short', 'tp_short'], inplace=True)
        
        self._signals = signals
        logger.info(f"Signaux générés : {entry_long.sum()} signaux d'achat, {entry_short.sum()} signaux de vente.")
