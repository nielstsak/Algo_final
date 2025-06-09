# src/backtesting/vectorbt_engine.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings

from src.core.config import settings
from src.core.constants import Trading, Kline
from src.core.exceptions import BacktestError, BacktestSetupError
from src.data.enriched_dataframe import EnrichedDataFrame
from src.backtesting.signal_adapter import SignalAdapter
from src.backtesting.fee_calculator import BinanceFeeCalculator
from src.backtesting.slippage_model import SlippageModel, FixedSlippageModel, VolumeBasedSlippageModel

class VectorBTEngine:
    """
    Moteur de backtesting utilisant VectorBT.
    Gère l'exécution des backtests avec support pour les frais, le slippage,
    et les paramètres réalistes de trading.
    Adapté pour utiliser un EnrichedDataFrame pour les backtests multi-fréquences.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        freq: Optional[str] = "1m",
        trade_on_close: bool = False,
        allow_shorting: bool = True,
        size_type: str = 'percent',
        default_size: float = 0.1
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.freq = freq
        self.trade_on_close = trade_on_close
        self.allow_shorting = allow_shorting
        self.size_type = size_type
        self.default_size = default_size
        
        self.fixed_commission_rate_from_cli = commission
        self.fixed_slippage_rate_from_cli = slippage

        if self.fixed_commission_rate_from_cli is None:
            self.fee_calculator_instance: Optional[BinanceFeeCalculator] = BinanceFeeCalculator()
            logger.info("Using dynamic BinanceFeeCalculator.")
        else:
            self.fee_calculator_instance = None
            logger.info(f"Using fixed commission rate from CLI: {self.fixed_commission_rate_from_cli*100:.4f}%")

        if self.fixed_slippage_rate_from_cli is None:
            self.slippage_model_instance: Optional[SlippageModel] = FixedSlippageModel()
            logger.info(f"Using dynamic {type(self.slippage_model_instance).__name__}.")
        else:
            self.slippage_model_instance = None
            logger.info(f"Using fixed slippage rate from CLI: {self.fixed_slippage_rate_from_cli*100:.4f}%")
            
        self.signal_adapter = SignalAdapter()
        
        self._last_portfolio: Optional[vbt.Portfolio] = None
        self._last_stats: Optional[pd.Series] = None
        
        logger.info(
            f"VectorBTEngine initialized: capital={initial_capital}, "
            f"leverage={leverage}, margin_mode={margin_mode}, execution_freq='{self.freq}'"
        )
        
    def _calculate_fees(self, data: pd.DataFrame, symbol: str) -> Union[float, pd.Series]:
        if self.fixed_commission_rate_from_cli is not None:
            return self.fixed_commission_rate_from_cli
        if self.fee_calculator_instance:
            is_maker = not self.trade_on_close
            return self.fee_calculator_instance.calculate_fees(data, symbol, is_maker=is_maker)
        return 0.001

    def _calculate_slippage(self, data: pd.DataFrame, symbol: str) -> Union[float, pd.Series]:
        if self.fixed_slippage_rate_from_cli is not None:
            return self.fixed_slippage_rate_from_cli
        if self.slippage_model_instance:
            return self.slippage_model_instance.calculate_slippage(data, symbol, trade_on_close=self.trade_on_close)
        return 0.0001

    def _validate_inputs(self, data: Union[pd.DataFrame, EnrichedDataFrame], signals: pd.DataFrame):
        if isinstance(data, EnrichedDataFrame):
            if data.df.empty:
                raise BacktestSetupError("EnrichedDataFrame is empty.")
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise BacktestSetupError("Input DataFrame is empty.")
            required_data_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_data_cols):
                raise BacktestSetupError(f"Missing required OHLC columns in DataFrame. Found: {list(data.columns)}")
        else:
            raise BacktestSetupError("Data must be a pandas DataFrame or EnrichedDataFrame.")

        if signals.empty and not (isinstance(data, EnrichedDataFrame) and data.df.empty):
             raise BacktestSetupError("Signals DataFrame is empty while data is not.")


    def _prepare_portfolio_params(
        self,
        data: pd.DataFrame, 
        entries: pd.Series,
        exits: pd.Series,
        size: Optional[Union[float, pd.Series]], 
        fees: Union[float, pd.Series],
        slippage: Union[float, pd.Series],
        short_entries: Optional[pd.Series] = None,
        short_exits: Optional[pd.Series] = None,
        sl_stop: Optional[Union[float, pd.Series]] = None, 
        tp_stop: Optional[Union[float, pd.Series]] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        params = {
            'close': data['close'], 'open': data['open'], 'high': data['high'], 'low': data['low'],     
            'entries': entries, 'exits': exits, 'short_entries': short_entries, 'short_exits': short_exits,
            'size': size if size is not None else self.default_size, 'size_type': self.size_type, 
            'fees': fees, 'slippage': slippage, 'init_cash': self.initial_capital, 'freq': self.freq, 
            'direction': 'both' if self.allow_shorting else 'longonly', 'accumulate': False, 
            'sl_stop': sl_stop, 'tp_stop': tp_stop, 
        }
        params.update(kwargs)
        return params

    def run_backtest(
        self,
        data: Union[pd.DataFrame, EnrichedDataFrame], 
        signals: pd.DataFrame, 
        symbol: str,
        **kwargs 
    ) -> vbt.Portfolio:
        try:
            self._validate_inputs(data, signals)
            
            if isinstance(data, EnrichedDataFrame):
                exec_data = data.get_view('1m')
                self.freq = '1m'
                logger.info(f"Running backtest with EnrichedDataFrame. Execution data extracted for '1m' frequency.")
                
                signals_propagated = signals.reindex(exec_data.index)
                bool_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short']
                for col in bool_cols:
                    if col in signals_propagated.columns:
                        signals_propagated[col] = signals_propagated[col].fillna(False).astype(bool)
                
                for col in ['sl', 'tp', 'size']:
                     if col in signals_propagated.columns:
                         signals_propagated[col] = signals_propagated[col].ffill()

                logger.info(f"Signals (freq: {signals.index.freqstr if hasattr(signals.index, 'freqstr') else 'inferred'}) propagated to 1m frequency as single pulses.")

            else:
                exec_data = data
                signals_propagated = signals
                logger.info("Running backtest with standard DataFrame.")

            entries = signals_propagated.get('entry_long', pd.Series(False, index=exec_data.index)).copy()
            exits = signals_propagated.get('exit_long', pd.Series(False, index=exec_data.index)).copy()
            short_entries = signals_propagated.get('entry_short', pd.Series(False, index=exec_data.index)).copy() if self.allow_shorting else pd.Series(False, index=exec_data.index)
            short_exits = signals_propagated.get('exit_short', pd.Series(False, index=exec_data.index)).copy() if self.allow_shorting else pd.Series(False, index=exec_data.index)

            if self.allow_shorting and self.size_type == 'percent':
                logger.debug("Nettoyage des signaux pour éviter les renversements de position non supportés...")
                
                # Étape 1: Identifier les signaux qui feraient un renversement direct de position
                # Un renversement serait: sortie long + entrée short sur la même bougie, ou sortie short + entrée long
                long_to_short_reversal = exits & short_entries
                short_to_long_reversal = short_exits & entries
                
                # Étape 2: Modifier les signaux pour éviter les renversements
                # Pour les renversements, nous allons d'abord fermer la position, puis entrer
                # dans la nouvelle position à la bougie suivante
                
                # Désactiver les entrées inverses sur les bougies de sortie
                short_entries.loc[long_to_short_reversal] = False
                entries.loc[short_to_long_reversal] = False
                
                # Reporter ces entrées à la bougie suivante
                postponed_short_entries = long_to_short_reversal.shift(1).fillna(False).infer_objects(copy=False)
                postponed_long_entries = short_to_long_reversal.shift(1).fillna(False).infer_objects(copy=False)
                
                # Ajouter les entrées reportées
                short_entries = short_entries | postponed_short_entries
                entries = entries | postponed_long_entries
                
                # Assurer qu'il n'y a pas de conflits entre entrées/sorties de même direction
                entries = entries & ~exits
                short_entries = short_entries & ~short_exits
            
            size_input = signals_propagated.get('size', None) 
            sl_stop = signals_propagated.get('sl', None) 
            tp_stop = signals_propagated.get('tp', None)

            fees_rate = self._calculate_fees(exec_data, symbol)
            slippage_rate = self._calculate_slippage(exec_data, symbol)
            
            portfolio_params = self._prepare_portfolio_params(
                exec_data, entries, exits, size_input, fees_rate, slippage_rate,
                short_entries=short_entries, short_exits=short_exits,
                sl_stop=sl_stop, tp_stop=tp_stop,
                **kwargs
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning) 
                portfolio = vbt.Portfolio.from_signals(**portfolio_params)
            
            self._last_portfolio = portfolio
            
            logger.info(f"Backtest completed for {symbol}. Trades: {portfolio.trades.count()}. Final Val: {portfolio.value().iloc[-1]:.2f}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}", exc_info=True)
            raise BacktestError(f"Failed to run backtest for {symbol}: {e}", original_exception=e)

    def run_multiple_backtests(
        self,
        data: Union[pd.DataFrame, EnrichedDataFrame],
        signals_dict: Dict[str, pd.DataFrame], 
        symbol: str,
        compare: bool = True
    ) -> Dict[str, vbt.Portfolio]:
        results = {}
        for name, signals_df in signals_dict.items():
            logger.info(f"Running backtest for variant: {name} on {symbol}")
            try:
                portfolio = self.run_backtest(data, signals_df, symbol)
                results[name] = portfolio
            except Exception as e:
                logger.error(f"Backtest failed for {name} on {symbol}: {e}")
        
        if compare and len(results) > 1:
            self._generate_comparison(results)
        return results

    def _generate_comparison(self, portfolios: Dict[str, vbt.Portfolio]):
        comparison_data = []
        first_pf_key = next(iter(portfolios), None)
        symbol_for_log = portfolios[first_pf_key].symbol if first_pf_key and hasattr(portfolios[first_pf_key], 'symbol') else 'N/A'

        for name, portfolio in portfolios.items():
            stats = portfolio.stats(settings=dict(risk_free_rate=0.0)) if portfolio.trades.count() > 0 else pd.Series(dtype=float)
            comparison_data.append({
                'Strategy/Variant': name,
                'Total Return [%]': stats.get('Total Return [%]', np.nan),
                'Sharpe Ratio': stats.get('Sharpe Ratio', np.nan),
                'Max Drawdown [%]': stats.get('Max Drawdown [%]', np.nan),
                'Win Rate [%]': stats.get('Win Rate [%]', np.nan),
                'Total Trades': portfolio.trades.count() 
            })
        comparison_df = pd.DataFrame(comparison_data).set_index('Strategy/Variant')
        logger.info(f"Strategy Comparison for {symbol_for_log}:\n{comparison_df.to_string(float_format='%.2f')}")
        return comparison_df

    def get_last_results(self) -> Optional[Dict[str, Any]]:
        if self._last_portfolio is None:
            return None
        
        results = {
            'portfolio': self._last_portfolio,
            'stats': self._last_stats if self._last_stats is not None else pd.Series(dtype=float),
            'equity_curve': self._last_portfolio.value(), 
            'drawdown_curve': self._last_portfolio.drawdown(), 
        }
        if self._last_portfolio.trades.count() > 0:
            results['trades_records'] = self._last_portfolio.trades.records_readable 
        else:
            results['trades_records'] = pd.DataFrame()
        return results
