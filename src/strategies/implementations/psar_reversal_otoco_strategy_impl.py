# src/strategies/implementations/psar_reversal_otoco_strategy_impl.py
import logging # Standard logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# Attempt to import the correct BaseStrategy and other necessary modules
try:
    from ..base_strategy import BaseStrategy # Relative import for strategy loader
    from src.core.exceptions import InvalidStrategyParamsError, StrategyError # Correct path
    from src.core.constants import Trading, Kline # Correct path
    from src.utils.exchange_utils import ( # Import if specific exchange utils are needed
        adjust_precision,
        get_filter_value,
        get_precision_from_filter
    )
except ImportError as e:
    # Log the specific import error to help diagnose
    logging.getLogger(__name__).critical(
        "PsarReversalOtocoStrategy: CRITICAL - Failed to import BaseStrategy or other core modules. "
        "Ensure 'src' is in PYTHONPATH and all dependencies are installed. Error: %s", e
    )
    # Define a minimal dummy BaseStrategy ONLY IF ABSOLUTELY NECESSARY for the file to be parsable.
    # This indicates a deeper setup problem that needs fixing.
    if 'BaseStrategy' not in globals(): # Check if BaseStrategy was successfully imported
        from abc import ABC, abstractmethod
        class BaseStrategy(ABC): # type: ignore
            name: str = "DummyBaseForPsar"
            version: str = "0.0.0"
            description: str = "Dummy BaseStrategy (Import Failed)"
            default_params: Dict[str, Any] = {}
            required_timeframes: List[str] = []
            min_required_periods: int = 1

            def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs): 
                self.params = self.default_params.copy()
                if params: self.params.update(params)
                self.params.update(kwargs)

            def validate_params(self) -> None: # Dummy implementation
                pass 
            
            def get_param(self, key: str, default: Any = None) -> Any: # Dummy get_param
                return self.params.get(key, default)

            @abstractmethod
            def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]: raise NotImplementedError
            @abstractmethod
            def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame: raise NotImplementedError
            def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None
            def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None
    
        # Define dummy exceptions if they also failed to import
        if 'InvalidStrategyParamsError' not in globals():
            class InvalidStrategyParamsError(Exception): pass # type: ignore
        if 'StrategyError' not in globals():
            class StrategyError(Exception): pass # type: ignore
        if 'get_precision_from_filter' not in globals(): # Dummy for exchange_utils
            def get_precision_from_filter(symbol_info: dict, filter_type: str, key: str) -> Optional[int]: return 8
        if 'adjust_precision' not in globals():
            def adjust_precision(value: float, precision: int, rounding_method = round) -> Optional[float]: return round(value, precision) if value is not None else None
        if 'get_filter_value' not in globals():
            def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None


# Use standard logging
logger = logging.getLogger(__name__)

class PsarReversalOtocoStrategy(BaseStrategy):
    name: str = "PsarReversalOtocoStrategy"
    version: str = "1.0.2" # Updated version
    description: str = "Stratégie basée sur le renversement du PSAR avec ordres OTOCO pour SL/TP."

    default_params: Dict[str, Any] = {
        'psar_step': 0.02,
        'psar_max_step': 0.2,
        'indicator_frequency': '1h', # Fréquence pour le calcul du PSAR
        'stop_loss_pct': 0.015,  # 1.5% stop loss
        'take_profit_pct': 0.03, # 3% take profit
        'position_sizing_pct_capital': 0.02, # Risque 2% du capital par trade
    }

    required_timeframes: List[str] = ['1m'] # La stratégie peut resampler en interne si besoin
    min_required_periods: int = 20 # Le PSAR a besoin de quelques périodes pour s'initialiser

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.pair_symbol = kwargs.get('pair_symbol', 'PAIR_UNSPECIFIED')
        super().__init__(params, **kwargs) # Appel à BaseStrategy.__init__ qui appelle self.validate_params()
        self.strategy_name_log_prefix = f"[{self.name}][{self.pair_symbol}]"

        psar_freq = self.get_param('indicator_frequency')
        psar_step_param = self.get_param('psar_step')
        psar_max_step_param = self.get_param('psar_max_step')
        # Construire un nom de colonne PSAR qui inclut les paramètres pour unicité
        self.psar_col = f"PSAR_{psar_freq}_s{str(psar_step_param).replace('.', '_')}_ms{str(psar_max_step_param).replace('.', '_')}"
        
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée.")
        logger.debug(f"{self.strategy_name_log_prefix} Expected PSAR column: '{self.psar_col}'")

    def validate_params(self) -> None:
        """Valide les paramètres de la stratégie."""
        psar_step = self.get_param('psar_step')
        psar_max_step = self.get_param('psar_max_step')
        sl_pct = self.get_param('stop_loss_pct')
        tp_pct = self.get_param('take_profit_pct')
        indicator_freq = self.get_param('indicator_frequency')

        if not (isinstance(psar_step, float) and 0 < psar_step < 1):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="psar_step", details="Must be a float between 0 and 1.")
        if not (isinstance(psar_max_step, float) and 0 < psar_max_step < 1):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="psar_max_step", details="Must be a float between 0 and 1.")
        if psar_step >= psar_max_step:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="psar_step/psar_max_step", details="psar_step must be less than psar_max_step.")
        if not (isinstance(sl_pct, float) and 0 < sl_pct < 1):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="stop_loss_pct", details="Must be a float between 0 and 1 (exclusive).")
        if not (isinstance(tp_pct, float) and 0 < tp_pct < 1):
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name="take_profit_pct", details="Must be a float between 0 and 1 (exclusive).")
        
        valid_freqs = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'] # Example
        if indicator_freq not in valid_freqs:
            raise InvalidStrategyParamsError(strategy_name=self.name, parameter_name='indicator_frequency', details=f"Invalid frequency '{indicator_freq}'. Supported: {valid_freqs}")
        
        logger.debug(f"{self.strategy_name_log_prefix} Parameters validated.")

    def calculate_indicators(self, klines: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        S'attend à ce que le PSAR soit pré-calculé et présent dans le DataFrame
        correspondant à 'indicator_frequency'.
        """
        log_pref = self.strategy_name_log_prefix
        primary_indicator_freq = self.get_param('indicator_frequency')

        if primary_indicator_freq not in klines or klines[primary_indicator_freq] is None:
            raise StrategyError(
                f"{log_pref} DataFrame for primary indicator frequency '{primary_indicator_freq}' not provided or None.",
                strategy_name=self.name
            )
        
        df_primary = klines[primary_indicator_freq].copy()
        logger.debug(f"{log_pref} Using data for timeframe '{primary_indicator_freq}'. Columns: {list(df_primary.columns)}")

        ohlc_map = {'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'}
        for src, target in ohlc_map.items():
            if src in df_primary.columns and target not in df_primary.columns:
                df_primary.rename(columns={src: target}, inplace=True)
            if target in df_primary.columns:
                 df_primary[target] = pd.to_numeric(df_primary[target], errors='coerce')

        if self.psar_col not in df_primary.columns:
            logger.warning(
                f"{log_pref} Expected PSAR column '{self.psar_col}' missing in DataFrame for '{primary_indicator_freq}'. "
                f"Signal generation might fail or be incorrect."
            )
            # Do not add NaN column here, let generate_signals handle it.

        self._indicators_cache = {primary_indicator_freq: df_primary}
        for tf, df_tf in klines.items(): # Pass through other TFs
            if tf != primary_indicator_freq and tf not in self._indicators_cache:
                self._indicators_cache[tf] = df_tf.copy()
        
        logger.debug(f"{log_pref} Indicators prepared. Main DF columns: {list(df_primary.columns)}")
        return self._indicators_cache

    def generate_signals(self, indicators: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Génère les signaux de trading basés sur le renversement du PSAR."""
        log_pref = self.strategy_name_log_prefix
        primary_indicator_freq = self.get_param('indicator_frequency')

        if primary_indicator_freq not in indicators or indicators[primary_indicator_freq] is None:
            logger.error(f"{log_pref} DataFrame for '{primary_indicator_freq}' not in indicators dict.")
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).astype(bool)

        df = indicators[primary_indicator_freq].copy()
        logger.debug(f"{log_pref} Generating signals on data from '{primary_indicator_freq}'. Shape: {df.shape}")

        required_cols = ['close', 'high', 'low', self.psar_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"{log_pref} Missing columns for signal generation: {missing_cols}. Returning empty signals.")
            return pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).fillna(False)

        if df.empty or any(df[col].isnull().all() for col in required_cols) or len(df) < 2:
            logger.warning(f"{log_pref} Insufficient data or all-NaN essential columns. Shape: {df.shape}")
            return pd.DataFrame(index=df.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']).fillna(False)

        psar = df[self.psar_col]
        close_price = df['close']
        # high_price = df['high'] # Not directly used in this simplified PSAR reversal logic for entry
        # low_price = df['low']   # Not directly used in this simplified PSAR reversal logic for entry

        # Conditions de renversement du PSAR
        # PSAR en dessous du prix -> tendance haussière potentielle
        # PSAR au-dessus du prix -> tendance baissière potentielle
        
        # Signal d'achat: PSAR passe de dessus à dessous le prix (ou le prix croise au-dessus du PSAR)
        # Pour simplifier: si PSAR(t-1) > close(t-1) ET PSAR(t) < close(t)
        # Ou, plus commun: si close(t) > PSAR(t) ET close(t-1) < PSAR(t-1) (prix croise PSAR)
        
        # Pour vectorbt, on a besoin de signaux d'entrée et de sortie.
        # Un renversement simple peut être traité comme: exit short & entry long, ou exit long & entry short.
        
        # Position du PSAR par rapport au prix
        psar_below_price = psar < close_price # True si PSAR est en dessous du prix (support haussier)
        # psar_above_price = psar > close_price # True si PSAR est au-dessus du prix (résistance baissière)

        # Détection des renversements
        # Renversement haussier (entry_long): PSAR était au-dessus, maintenant en dessous
        entry_long_cond = psar_below_price & ~psar_below_price.shift(1).fillna(False)
        
        # Renversement baissier (entry_short): PSAR était en dessous, maintenant au-dessus
        entry_short_cond = ~psar_below_price & psar_below_price.shift(1).fillna(True) # ~psar_below_price = psar_above_price (ou égal)

        signals_df = pd.DataFrame(index=df.index)
        signals_df['entry_long'] = entry_long_cond
        signals_df['entry_short'] = entry_short_cond

        # Les sorties sont déclenchées par le renversement opposé
        signals_df['exit_long'] = entry_short_cond  # Sortir d'un long si un signal short apparaît
        signals_df['exit_short'] = entry_long_cond # Sortir d'un short si un signal long apparaît

        # Calcul du SL/TP basé sur pourcentage
        sl_pct = self.get_param('stop_loss_pct')
        tp_pct = self.get_param('take_profit_pct')
        
        entry_price_series = df['close'] # Utiliser le 'close' comme prix d'entrée de référence pour SL/TP

        signals_df['sl'] = np.where(
            signals_df['entry_long'], entry_price_series * (1 - sl_pct),
            np.where(signals_df['entry_short'], entry_price_series * (1 + sl_pct), np.nan)
        )
        signals_df['tp'] = np.where(
            signals_df['entry_long'], entry_price_series * (1 + tp_pct),
            np.where(signals_df['entry_short'], entry_price_series * (1 - tp_pct), np.nan)
        )
        
        for col_b in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            signals_df[col_b] = signals_df[col_b].fillna(False).astype(bool)

        self._signals = signals_df
        logger.debug(f"{log_pref} Signaux générés. Longs: {signals_df['entry_long'].sum()}, Shorts: {signals_df['entry_short'].sum()}")
        return self._signals.copy()

    def generate_order_request(self,
                               data_dict: Dict[str, pd.DataFrame],
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre pour le trading live.
        """
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        primary_indicator_freq = self.get_param('indicator_frequency')

        if primary_indicator_freq not in data_dict or data_dict[primary_indicator_freq] is None or data_dict[primary_indicator_freq].empty:
            logger.warning(f"{log_prefix_live} Données pour timeframe '{primary_indicator_freq}' manquantes ou vides.")
            return None
            
        df_indicators = data_dict[primary_indicator_freq]
        logger.info(f"{log_prefix_live} Appel generate_order_request. Position: {current_position}, Capital: {available_capital:.2f}")

        if len(df_indicators) < 2: # Besoin d'au moins deux points pour détecter un renversement
            logger.warning(f"{log_prefix_live} Données d'indicateurs insuffisantes (lignes: {len(df_indicators)}). Pas d'ordre.")
            return None

        latest_data = df_indicators.iloc[-1]
        previous_data = df_indicators.iloc[-2]

        required_cols_live = ['close', 'high', 'low', self.psar_col] # high/low pour PSAR
        if latest_data[required_cols_live].isnull().any() or \
           previous_data[required_cols_live].isnull().any():
            logger.warning(f"{log_prefix_live} Indicateurs NaN sur dernières données. Pas d'ordre.")
            return None

        psar_curr = latest_data[self.psar_col]
        close_curr = latest_data['close']
        psar_prev = previous_data[self.psar_col]
        # close_prev = previous_data['close'] # Non utilisé directement dans la logique simplifiée ci-dessous

        side: Optional[str] = None
        entry_price_for_order = close_curr

        # Logique de signal pour la dernière barre
        psar_below_price_curr = psar_curr < close_curr
        psar_below_price_prev = psar_prev < previous_data['close'] # Utiliser close_prev ici

        if current_position == 0: # Logique d'entrée
            if psar_below_price_curr and not psar_below_price_prev: # Renversement haussier
                side = Trading.SIDE_BUY
                logger.info(f"{log_prefix_live} Signal ACHAT (entrée). PSAR {psar_curr:.5f} < Close {close_curr:.5f}. Prev PSAR {psar_prev:.5f} >= Prev Close {previous_data['close']:.5f}")
            elif not psar_below_price_curr and psar_below_price_prev: # Renversement baissier
                side = Trading.SIDE_SELL
                logger.info(f"{log_prefix_live} Signal VENTE (entrée). PSAR {psar_curr:.5f} >= Close {close_curr:.5f}. Prev PSAR {psar_prev:.5f} < Prev Close {previous_data['close']:.5f}")
        
        elif current_position > 0: # Actuellement LONG, chercher sortie
            if not psar_below_price_curr and psar_below_price_prev: # Renversement baissier (sortie LONG)
                side = Trading.SIDE_SELL # Ordre de vente pour fermer le long
                logger.info(f"{log_prefix_live} Signal SORTIE LONG. PSAR {psar_curr:.5f} >= Close {close_curr:.5f}. Prev PSAR {psar_prev:.5f} < Prev Close {previous_data['close']:.5f}")
        
        elif current_position < 0: # Actuellement SHORT, chercher sortie
            if psar_below_price_curr and not psar_below_price_prev: # Renversement haussier (sortie SHORT)
                side = Trading.SIDE_BUY # Ordre d'achat pour fermer le short
                logger.info(f"{log_prefix_live} Signal SORTIE SHORT. PSAR {psar_curr:.5f} < Close {close_curr:.5f}. Prev PSAR {psar_prev:.5f} >= Prev Close {previous_data['close']:.5f}")


        if side:
            sl_pct = self.get_param('stop_loss_pct')
            tp_pct = self.get_param('take_profit_pct')
            
            # Pour une sortie, le SL/TP n'est pas pertinent de la même manière qu'une entrée.
            # L'ordre de sortie est un ordre MARKET ou LIMIT pour fermer la position.
            if current_position != 0 : # Si c'est un signal de sortie
                # Calculer la quantité pour fermer la position (logique simplifiée)
                # La quantité exacte dépendra de la gestion de position de l'engine.
                # Ici, on signale juste l'intention de fermer.
                # Le trading engine devrait déterminer la quantité exacte à fermer.
                # Pour l'instant, on ne retourne pas de SL/TP pour les ordres de sortie.
                
                # Exemple: La quantité à fermer est la taille de la position actuelle.
                # Ceci est une simplification, le trading engine gère la taille de la position.
                # Nous ne pouvons pas la calculer ici sans connaître la taille exacte de la position ouverte.
                # Pour un signal de sortie, la quantité est souvent implicite (fermer tout).
                # Le TradingEngine devrait gérer cela.
                # Ici, on peut retourner une quantité symbolique ou laisser le TradingEngine décider.
                dummy_qty_to_close = 0.001 # Placeholder, le TradingEngine doit gérer la taille de la position
                
                price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize') or 8
                qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8

                quantity_str = f"{dummy_qty_to_close:.{qty_precision}f}" # Doit être la taille de la position actuelle
                # Pour un ordre de sortie MARKET, le prix n'est pas nécessaire.
                # Pour un ordre LIMIT, on pourrait utiliser le close_curr ou un prix légèrement agressif.
                entry_price_str = f"{entry_price_for_order:.{price_precision}f}"


                order_request_params = {
                    "symbol": symbol, "side": side, "type": "MARKET", # Sortie au marché
                    "quantity": quantity_str, # Le TradingEngine doit remplacer par la taille de la position
                }
                logger.info(f"{log_prefix_live} Requête d'ordre de SORTIE générée: {order_request_params}")
                return order_request_params, {} # Pas de SL/TP pour un ordre de fermeture simple


            # Logique pour les ordres d'ENTRÉE (current_position == 0)
            if side == Trading.SIDE_BUY:
                stop_loss_price_raw = entry_price_for_order * (1 - sl_pct)
                take_profit_price_raw = entry_price_for_order * (1 + tp_pct)
            else: # SELL
                stop_loss_price_raw = entry_price_for_order * (1 + sl_pct)
                take_profit_price_raw = entry_price_for_order * (1 - tp_pct)

            price_precision = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize') or 8
            qty_precision = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize') or 8

            # Calcul de la quantité (exemple simplifié)
            capital_to_risk_trade = available_capital * self.get_param('position_sizing_pct_capital')
            risk_per_unit = abs(entry_price_for_order - stop_loss_price_raw)
            if risk_per_unit < 1e-9: # Eviter division par zéro ou risque infini
                logger.warning(f"{log_prefix_live} Risque par unité trop faible ou nul. Pas d'ordre. Entrée: {entry_price_for_order}, SL: {stop_loss_price_raw}")
                return None
            quantity_raw = capital_to_risk_trade / risk_per_unit
            
            quantity_adjusted = adjust_precision(quantity_raw, qty_precision, rounding_method=np.floor)
            if quantity_adjusted is None or quantity_adjusted <= 0:
                logger.warning(f"{log_prefix_live} Quantité ajustée invalide ({quantity_adjusted}). Pas d'ordre.")
                return None
            quantity_str = f"{quantity_adjusted:.{qty_precision}f}"
            
            entry_price_adjusted = adjust_precision(entry_price_for_order, price_precision, round)
            if entry_price_adjusted is None:
                logger.error(f"{log_prefix_live} Échec de l'ajustement du prix d'entrée. Pas d'ordre.")
                return None
            entry_price_str = f"{entry_price_adjusted:.{price_precision}f}"

            sl_price_final = adjust_precision(stop_loss_price_raw, price_precision, round)
            tp_price_final = adjust_precision(take_profit_price_raw, price_precision, round)

            if sl_price_final is None or tp_price_final is None:
                logger.error(f"{log_prefix_live} Échec de l'ajustement SL/TP. Pas d'ordre.")
                return None

            order_request_params = {
                "symbol": symbol, "side": side, "type": "LIMIT", # Ou MARKET
                "quantity": quantity_str, "price": entry_price_str,
            }
            sl_tp_for_engine = {'sl_price': sl_price_final, 'tp_price': tp_price_final}
            logger.info(f"{log_prefix_live} Requête d'ordre d'ENTRÉE générée: {order_request_params} avec SL/TP: {sl_tp_for_engine}")
            return order_request_params, sl_tp_for_engine
        
        logger.debug(f"{log_prefix_live} Aucune condition d'entrée/sortie remplie pour un ordre live.")
        return None

