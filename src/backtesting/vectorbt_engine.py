# src/backtesting/vectorbt_engine.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings

from src.core.config import settings
from src.core.constants import Trading, Kline # Assuming Kline might be used for freq defaults
from src.core.exceptions import BacktestError, BacktestSetupError
from src.backtesting.signal_adapter import SignalAdapter
from src.backtesting.fee_calculator import BinanceFeeCalculator # Or your base FeeCalculator
# MODIFICATION ICI: Ajout de SlippageModel à l'import
from src.backtesting.slippage_model import SlippageModel, FixedSlippageModel, VolumeBasedSlippageModel 

class VectorBTEngine:
    """
    Moteur de backtesting utilisant VectorBT.
    Gère l'exécution des backtests avec support pour les frais, le slippage,
    et les paramètres réalistes de trading.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: Optional[float] = None, # Fixed commission rate from CLI
        slippage: Optional[float] = None,   # Fixed slippage rate from CLI
        leverage: float = 1.0,
        margin_mode: str = "cross",
        freq: Optional[str] = None, # e.g., '1h', '1d'. VectorBT will parse this.
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
            # Type hint pour l'instance du modèle de slippage
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
            f"leverage={leverage}, margin_mode={margin_mode}, freq='{self.freq}'"
        )
        
    def _calculate_fees(
        self,
        data: pd.DataFrame, 
        symbol: str
    ) -> Union[float, pd.Series]:
        if self.fixed_commission_rate_from_cli is not None:
            return self.fixed_commission_rate_from_cli
        
        if self.fee_calculator_instance:
            is_maker = not self.trade_on_close 
            return self.fee_calculator_instance.calculate_fees(data, symbol, is_maker=is_maker)
        
        logger.warning("No fee model or fixed rate defined, defaulting to 0.1% fees.")
        return 0.001

    def _calculate_slippage(
        self,
        data: pd.DataFrame, 
        symbol: str
    ) -> Union[float, pd.Series]:
        if self.fixed_slippage_rate_from_cli is not None:
            return self.fixed_slippage_rate_from_cli
            
        if self.slippage_model_instance:
            return self.slippage_model_instance.calculate_slippage(
                data, 
                symbol, 
                trade_on_close=self.trade_on_close
            )

        logger.warning("No slippage model or fixed rate defined, defaulting to 0.01% slippage.")
        return 0.0001

    def _validate_inputs(self, data: pd.DataFrame, signals: pd.DataFrame):
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise BacktestSetupError("Data must be a non-empty pandas DataFrame.")
        if not isinstance(signals, pd.DataFrame) or signals.empty:
            # Allow empty signals if data is also empty (e.g. no data for period)
            if not data.empty:
                 raise BacktestSetupError("Signals must be a non-empty pandas DataFrame if data is present.")
            else: # Both empty, this is okay, backtest will be trivial
                logger.info("Data and signals are both empty. Backtest will be trivial.")
                return


        required_data_cols = ['open', 'high', 'low', 'close'] 
        missing_cols = [col for col in required_data_cols if col not in data.columns]
        if missing_cols:
            raise BacktestSetupError(f"Missing required columns in data: {missing_cols}. Expected lowercase OHLC.")
            
        if not data.index.equals(signals.index) and not (data.empty or signals.empty):
            logger.warning("Data and signals indices do not match perfectly. Attempting to align signals to data index.")
            try:
                # Prioritize data index, reindex signals, ffill for entries/exits, specific fill for others
                original_signal_columns = signals.columns.tolist()
                signals = signals.reindex(data.index)
                for col in original_signal_columns:
                    if 'entry' in col or 'exit' in col: # Boolean signals
                        signals[col] = signals[col].fillna(False)
                    # For 'sl', 'tp', 'size', NaN is often appropriate if no signal at that point
                    # Default reindex behavior (NaN fill) is usually fine for these numeric/optional columns
                logger.info("Successfully reindexed signals to match data index.")
            except Exception as e:
                raise BacktestSetupError(f"Failed to align signals index with data index: {e}")


        for col in ['entries', 'exits', 'short_entries', 'short_exits']:
            if col in signals.columns and signals[col].notna().any() and not pd.api.types.is_bool_dtype(signals[col]):
                logger.warning(f"Signal column '{col}' is not boolean. Attempting to convert.")
                try:
                    signals[col] = signals[col].astype(bool)
                except Exception as e:
                    raise BacktestSetupError(f"Failed to convert signal column '{col}' to boolean: {e}")

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
            'close': data['close'], 
            'open': data['open'],   
            'high': data['high'],   
            'low': data['low'],     
            'entries': entries,
            'exits': exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'size': size if size is not None else self.default_size, 
            'size_type': self.size_type, 
            'fees': fees,
            'slippage': slippage,
            'init_cash': self.initial_capital,
            'freq': self.freq, 
            'direction': 'both' if self.allow_shorting else 'longonly',
            'accumulate': False, 
            'sl_stop': sl_stop, 
            'tp_stop': tp_stop, 
            'trade_on_close': self.trade_on_close, 
            'call_seq': 'slbtp' if (sl_stop is not None or tp_stop is not None) else 'default',
        }
        
        params.update(kwargs)
        # More selective logging for large series
        loggable_params = {}
        for k, v_item in params.items(): # Renamed v to v_item
            if isinstance(v_item, (pd.Series, pd.DataFrame)) and len(v_item) > 10:
                loggable_params[k] = f"{type(v_item).__name__}(len={len(v_item)})"
            elif isinstance(v_item, (pd.Series, pd.DataFrame)): # Short series
                 loggable_params[k] = f"{type(v_item).__name__}({v_item.to_dict() if isinstance(v_item, pd.Series) else 'DataFrame'})"
            else:
                loggable_params[k] = v_item
        logger.debug(f"Portfolio.from_signals params: {loggable_params}")
        return params

    def run_backtest(
        self,
        data: pd.DataFrame, 
        signals: pd.DataFrame, 
        symbol: str,
        **kwargs 
    ) -> vbt.Portfolio:
        try:
            self._validate_inputs(data, signals) # signals might be modified here if reindexed
            
            entries = signals.get('entries', pd.Series(False, index=data.index))
            exits = signals.get('exits', pd.Series(False, index=data.index))
            
            short_entries = signals.get('short_entries', None) if self.allow_shorting else None
            short_exits = signals.get('short_exits', None) if self.allow_shorting else None
            
            size_input_for_vbt = signals.get('size', None) 

            sl_stop_values = signals.get('sl', None) 
            tp_stop_values = signals.get('tp', None)

            fees_rate = self._calculate_fees(data, symbol)
            slippage_rate = self._calculate_slippage(data, symbol)
            
            portfolio_params = self._prepare_portfolio_params(
                data, entries, exits, size_input_for_vbt, fees_rate, slippage_rate,
                short_entries=short_entries, short_exits=short_exits,
                sl_stop=sl_stop_values, tp_stop=tp_stop_values,
                **kwargs
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning) 
                portfolio = vbt.Portfolio.from_signals(**portfolio_params)
            
            self._last_portfolio = portfolio
            if portfolio.trades.count() > 0:
                self._last_stats = portfolio.stats(settings=dict(risk_free_rate=0.0)) # Pass default risk_free_rate
            else:
                self._last_stats = pd.Series(dtype=float) 
            
            logger.info(f"Backtest completed for {symbol}. Trades: {portfolio.trades.count()}. Final Val: {portfolio.value().iloc[-1]:.2f}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}", exc_info=True)
            raise BacktestError(f"Failed to run backtest for {symbol}: {e}", original_exception=e)

    def run_multiple_backtests(
        self,
        data: pd.DataFrame,
        signals_dict: Dict[str, pd.DataFrame], 
        symbol: str,
        compare: bool = True
    ) -> Dict[str, vbt.Portfolio]:
        results = {}
        for name, signals_df in signals_dict.items():
            logger.info(f"Running backtest for variant: {name} on {symbol}")
            try:
                # Ensure data and signals are fresh copies for each run if they were modified
                portfolio = self.run_backtest(data.copy(), signals_df.copy(), symbol)
                results[name] = portfolio
            except Exception as e:
                logger.error(f"Backtest failed for {name} on {symbol}: {e}")
        
        if compare and len(results) > 1:
            self._generate_comparison(results)
        return results

    def _generate_comparison(self, portfolios: Dict[str, vbt.Portfolio]):
        comparison_data = []
        # Ensure there's at least one portfolio to get the symbol from
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
