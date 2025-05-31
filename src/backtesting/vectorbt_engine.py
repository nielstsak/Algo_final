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
from src.backtesting.signal_adapter import SignalAdapter
from src.backtesting.fee_calculator import BinanceFeeCalculator
from src.backtesting.slippage_model import SlippageModel


class VectorBTEngine:
    """
    Moteur de backtesting utilisant VectorBT.
    Gère l'exécution des backtests avec support pour les frais, le slippage,
    et les paramètres réalistes de trading.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        leverage: float = 1.0,
        margin_mode: str = "cross",
        freq: Optional[str] = None,
        trade_on_close: bool = False,
        allow_shorting: bool = True,
        size_type: str = 'percent',
        default_size: float = 0.1
    ):
        """
        Initialise le moteur de backtesting.
        
        Args:
            initial_capital: Capital initial
            commission: Commission (si None, utilise BinanceFeeCalculator)
            slippage: Slippage en pourcentage (si None, utilise SlippageModel)
            leverage: Levier maximum
            margin_mode: Mode de marge ("cross" ou "isolated")
            freq: Fréquence des données
            trade_on_close: Exécuter les trades à la clôture
            allow_shorting: Autoriser les positions courtes
            size_type: Type de taille ('percent', 'amount', 'value')
            default_size: Taille par défaut des positions
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.freq = freq
        self.trade_on_close = trade_on_close
        self.allow_shorting = allow_shorting
        self.size_type = size_type
        self.default_size = default_size
        
        # Calculateurs de frais et slippage
        self.fee_calculator = BinanceFeeCalculator() if commission is None else None
        self.slippage_model = SlippageModel() if slippage is None else None
        self.fixed_commission = commission
        self.fixed_slippage = slippage
        
        # Adaptateur de signaux
        self.signal_adapter = SignalAdapter()
        
        # Cache des résultats
        self._last_portfolio = None
        self._last_stats = None
        
        logger.info(
            f"VectorBTEngine initialized: capital={initial_capital}, "
            f"leverage={leverage}, margin_mode={margin_mode}"
        )
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        symbol: str,
        **kwargs
    ) -> 'vbt.Portfolio':
        """
        Exécute un backtest sur les données et signaux fournis.
        
        Args:
            data: DataFrame avec les données OHLCV
            signals: DataFrame avec les signaux de trading
            symbol: Symbole tradé
            **kwargs: Paramètres additionnels pour VectorBT
            
        Returns:
            Portfolio VectorBT avec les résultats
        """
        try:
            # Valider les entrées
            self._validate_inputs(data, signals)
            
            # Adapter les signaux au format VectorBT
            entries, exits, size = self.signal_adapter.adapt_signals(
                signals,
                allow_shorting=self.allow_shorting,
                size_type=self.size_type,
                default_size=self.default_size
            )
            
            # Calculer les frais et slippage
            fees = self._calculate_fees(data, symbol)
            slippage = self._calculate_slippage(data, symbol)
            
            # Configurer les paramètres du portfolio
            portfolio_params = self._prepare_portfolio_params(
                data, entries, exits, size, fees, slippage, **kwargs
            )
            
            # Exécuter le backtest
            portfolio = vbt.Portfolio.from_signals(**portfolio_params)
            
            # Sauvegarder les résultats
            self._last_portfolio = portfolio
            self._last_stats = portfolio.stats()
            
            logger.info(f"Backtest completed for {symbol}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise BacktestError(f"Failed to run backtest: {e}", original_exception=e)
            
    def _validate_inputs(self, data: pd.DataFrame, signals: pd.DataFrame):
        """Valide les données et signaux d'entrée."""
        if data.empty:
            raise BacktestSetupError("Empty data provided")
            
        if signals.empty:
            raise BacktestSetupError("Empty signals provided")
            
        # Vérifier l'alignement des index
        if not data.index.equals(signals.index):
            logger.warning("Data and signals indices don't match perfectly")
            
        # Vérifier les colonnes requises
        required_data_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_data_cols if col not in data.columns]
        if missing_cols:
            raise BacktestSetupError(f"Missing required columns in data: {missing_cols}")
            
    def _calculate_fees(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Union[float, pd.Series]:
        """
        Calcule les frais de trading.
        
        Returns:
            Frais fixes ou série de frais variables
        """
        if self.fixed_commission is not None:
            return self.fixed_commission
            
        if self.fee_calculator:
            return self.fee_calculator.calculate_fees(
                data,
                symbol,
                is_maker=not self.trade_on_close  # Limit orders are makers
            )
        
        # Par défaut, utiliser les frais Binance standards
        return 0.001  # 0.1%
        
    def _calculate_slippage(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Union[float, pd.Series]:
        """
        Calcule le slippage.
        
        Returns:
            Slippage fixe ou série de slippage variable
        """
        if self.fixed_slippage is not None:
            return self.fixed_slippage
            
        if self.slippage_model:
            return self.slippage_model.calculate_slippage(
                data,
                symbol,
                self.trade_on_close
            )
            
        # Par défaut, slippage minime
        return 0.0001  # 0.01%
        
    def _prepare_portfolio_params(
        self,
        data: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        size: pd.DataFrame,
        fees: Union[float, pd.Series],
        slippage: Union[float, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prépare les paramètres pour VectorBT Portfolio.
        
        Returns:
            Dict avec tous les paramètres configurés
        """
        params = {
            'close': data['close'],
            'entries': entries,
            'exits': exits,
            'size': size,
            'fees': fees,
            'slippage': slippage,
            'init_cash': self.initial_capital,
            'freq': self.freq,
            'direction': 'both' if self.allow_shorting else 'longonly',
            'accumulate': False,  # Ne pas accumuler les positions
            'sl_stop': None,  # Géré par les signaux
            'tp_stop': None,  # Géré par les signaux
            'call_seq': 'auto',  # Ordre d'exécution automatique
        }
        
        # Prix d'exécution
        if self.trade_on_close:
            params['price'] = data['close']
        else:
            params['price'] = data['open'].shift(-1)  # Prix d'ouverture suivant
            
        # Ajouter les paramètres personnalisés
        params.update(kwargs)
        
        return params
        
    def run_multiple_backtests(
        self,
        data: pd.DataFrame,
        signals_dict: Dict[str, pd.DataFrame],
        symbol: str,
        compare: bool = True
    ) -> Dict[str, 'vbt.Portfolio']:
        """
        Exécute plusieurs backtests pour comparer différentes stratégies.
        
        Args:
            data: DataFrame avec les données OHLCV
            signals_dict: Dict {strategy_name: signals_df}
            symbol: Symbole tradé
            compare: Si True, génère une comparaison
            
        Returns:
            Dict avec les portfolios par stratégie
        """
        results = {}
        
        for strategy_name, signals in signals_dict.items():
            logger.info(f"Running backtest for {strategy_name}")
            try:
                portfolio = self.run_backtest(data, signals, symbol)
                results[strategy_name] = portfolio
            except Exception as e:
                logger.error(f"Backtest failed for {strategy_name}: {e}")
                
        if compare and len(results) > 1:
            self._generate_comparison(results)
            
        return results
        
    def _generate_comparison(self, portfolios: Dict[str, 'vbt.Portfolio']):
        """Génère une comparaison entre plusieurs portfolios."""
        # Extraire les métriques clés
        comparison_data = []
        
        for name, portfolio in portfolios.items():
            stats = portfolio.stats()
            comparison_data.append({
                'Strategy': name,
                'Total Return [%]': stats.get('Total Return [%]', 0),
                'Sharpe Ratio': stats.get('Sharpe Ratio', 0),
                'Max Drawdown [%]': stats.get('Max Drawdown [%]', 0),
                'Win Rate [%]': stats.get('Win Rate [%]', 0),
                'Total Trades': stats.get('Total Trades', 0)
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Strategy', inplace=True)
        
        logger.info("Strategy Comparison:")
        logger.info(f"\n{comparison_df}")
        
        return comparison_df
        
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: callable,
        param_grid: Dict[str, List[Any]],
        symbol: str,
        metric: str = 'sharpe_ratio',
        n_jobs: int = -1
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Optimise les paramètres d'une stratégie.
        
        Args:
            data: DataFrame avec les données OHLCV
            strategy_func: Fonction qui génère les signaux
            param_grid: Grille de paramètres à tester
            symbol: Symbole tradé
            metric: Métrique à optimiser
            n_jobs: Nombre de jobs parallèles
            
        Returns:
            Tuple (best_params, results_df)
        """
        from itertools import product
        
        # Générer toutes les combinaisons
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        results = []
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            
            try:
                # Générer les signaux avec ces paramètres
                signals = strategy_func(data, **params)
                
                # Exécuter le backtest
                portfolio = self.run_backtest(data, signals, symbol)
                
                # Extraire la métrique
                stats = portfolio.stats()
                metric_value = self._extract_metric(stats, metric)
                
                result = params.copy()
                result[metric] = metric_value
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                
        # Créer le DataFrame des résultats
        results_df = pd.DataFrame(results)
        
        # Trouver les meilleurs paramètres
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx].to_dict()
        del best_params[metric]  # Retirer la métrique des paramètres
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {results_df.loc[best_idx, metric]}")
        
        return best_params, results_df
        
    def _extract_metric(self, stats: pd.Series, metric: str) -> float:
        """Extrait une métrique spécifique des statistiques."""
        metric_map = {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': 'Total Return [%]',
            'max_drawdown': 'Max Drawdown [%]',
            'win_rate': 'Win Rate [%]',
            'profit_factor': 'Profit Factor',
            'calmar_ratio': 'Calmar Ratio'
        }
        
        vbt_metric = metric_map.get(metric, metric)
        value = stats.get(vbt_metric, np.nan)
        
        # Inverser les métriques négatives pour l'optimisation
        if metric in ['max_drawdown']:
            value = -value
            
        return value
        
    def get_last_results(self) -> Optional[Dict[str, Any]]:
        """Retourne les derniers résultats de backtest."""
        if self._last_portfolio is None:
            return None
            
        return {
            'portfolio': self._last_portfolio,
            'stats': self._last_stats,
            'equity_curve': self._last_portfolio.value(),
            'drawdown': self._last_portfolio.drawdown(),
            'trades': self._last_portfolio.trades.records_arr
        }