import pandas as pd
import numpy as np
import vectorbt as vbt
from scipy.stats import norm
from typing import Dict, List, Optional, Any, Tuple, Union
from loguru import logger

from src.core.exceptions import BacktestError

# Constants for annualization
TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52
TRADING_MONTHS_PER_YEAR = 12

# Default risk-free rate for ratios if not specified
DEFAULT_RISK_FREE_RATE_ANNUAL = 0.0

class PerformanceMetrics:
    """
    Calculates various performance metrics for a backtest.
    This class primarily uses static methods to compute metrics from a
    vectorbt.Portfolio object or its components like returns and trades.
    """

    @staticmethod
    def _get_risk_free_rate_for_period(
        risk_free_rate_annual: float,
        returns_period: pd.Timedelta
    ) -> float:
        """
        Converts an annual risk-free rate to the rate for the given returns period.
        """
        if returns_period == pd.Timedelta(days=1):
            return (1 + risk_free_rate_annual)**(1/TRADING_DAYS_PER_YEAR) - 1
        elif returns_period == pd.Timedelta(weeks=1):
            return (1 + risk_free_rate_annual)**(1/TRADING_WEEKS_PER_YEAR) - 1
        elif returns_period.days >= 28 and returns_period.days <= 31: # Monthly
            return (1 + risk_free_rate_annual)**(1/TRADING_MONTHS_PER_YEAR) - 1
        else:
            days_in_period = returns_period.total_seconds() / (24 * 3600)
            if days_in_period > 0 and TRADING_DAYS_PER_YEAR / days_in_period > 0:
                return (1 + risk_free_rate_annual)**(days_in_period / TRADING_DAYS_PER_YEAR) - 1
            return 0.0

    # --- 1. Return Metrics ---
    @staticmethod
    def total_return(portfolio: vbt.Portfolio) -> float:
        """Calculates the total cumulative return of the strategy."""
        return portfolio.total_return()

    @staticmethod
    def annualized_return(portfolio: vbt.Portfolio) -> float:
        """Calculates the annualized return."""
        return portfolio.annualized_return()

    @staticmethod
    def daily_returns(portfolio: vbt.Portfolio) -> pd.Series:
        """Extracts daily returns. If portfolio frequency is not daily, it resamples."""
        if portfolio.wrapper.freq == pd.Timedelta(days=1):
            return portfolio.returns()
        else:
            daily_values = portfolio.value().resample('D').last().ffill()
            return daily_values.pct_change().dropna()

    @staticmethod
    def monthly_returns(portfolio: vbt.Portfolio) -> pd.Series:
        """Calculates monthly returns from portfolio values."""
        monthly_values = portfolio.value().resample('M').last()
        return monthly_values.pct_change().dropna()

    # --- 2. Risk Ratios ---
    @staticmethod
    def sharpe_ratio(
        portfolio: vbt.Portfolio,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs
    ) -> float:
        """Calculates the Sharpe Ratio."""
        return portfolio.sharpe_ratio(risk_free=risk_free_rate_annual, **kwargs)

    @staticmethod
    def sortino_ratio(
        portfolio: vbt.Portfolio,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs
    ) -> float:
        """Calculates the Sortino Ratio."""
        return portfolio.sortino_ratio(required_return=risk_free_rate_annual, **kwargs)

    @staticmethod
    def calmar_ratio(portfolio: vbt.Portfolio, **kwargs) -> float:
        """Calculates the Calmar Ratio."""
        return portfolio.calmar_ratio(**kwargs)

    @staticmethod
    def information_ratio(
        portfolio: vbt.Portfolio,
        benchmark_returns: pd.Series,
        **kwargs
    ) -> float:
        """Calculates the Information Ratio."""
        if benchmark_returns is None or benchmark_returns.empty:
            return np.nan
        try:
            return portfolio.information_ratio(benchmark_returns, **kwargs)
        except Exception:
            return np.nan

    # --- 3. Drawdown Metrics ---
    @staticmethod
    def max_drawdown(portfolio: vbt.Portfolio) -> float:
        """Calculates the Maximum Drawdown (MDD)."""
        return portfolio.max_drawdown()

    @staticmethod
    def max_drawdown_duration(portfolio: vbt.Portfolio) -> Union[pd.Timedelta, float]:
        """Calculates the duration of the longest drawdown period."""
        stats = portfolio.stats()
        duration_days = stats.get('Max Drawdown Duration', np.nan) 
        if pd.isna(duration_days):
            return np.nan
        return pd.to_timedelta(duration_days, unit='d')

    @staticmethod
    def underwater_curve(portfolio: vbt.Portfolio) -> pd.Series:
        """Returns the underwater curve (drawdowns over time)."""
        return portfolio.drawdown()

    @staticmethod
    def recovery_time(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """
        Calculates the time to recover from the maximum drawdown.
        """
        if portfolio.trades.count() == 0:
            return None
            
        drawdowns_df = portfolio.drawdowns.records_readable.copy()
        if drawdowns_df.empty:
            return None

        if 'Peak Value' in drawdowns_df.columns and 'Valley Value' in drawdowns_df.columns:
            peak_values = pd.to_numeric(drawdowns_df['Peak Value'], errors='coerce')
            valley_values = pd.to_numeric(drawdowns_df['Valley Value'], errors='coerce')
            
            drawdown_pct = (peak_values - valley_values) / peak_values.replace(0, np.nan)
            drawdowns_df['CalculatedDrawdown'] = drawdown_pct
            
            if drawdowns_df['CalculatedDrawdown'].notna().any():
                    mdd_record = drawdowns_df.loc[drawdowns_df['CalculatedDrawdown'].idxmax()]
            else:
                logger.warning("Impossible de calculer le drawdown à partir de Peak/Valley values.")
                return None
        else:
            logger.warning(f"Colonnes 'Peak Value' ou 'Valley Value' non trouvées pour calculer le drawdown. Colonnes: {list(drawdowns_df.columns)}")
            return None

        recovery_date = mdd_record.get('End Timestamp')
        valley_date = mdd_record.get('Valley Timestamp')
        
        if pd.isna(recovery_date) or pd.isna(valley_date):
            return None 

        recovery_duration_td = pd.to_datetime(recovery_date) - pd.to_datetime(valley_date)
        return recovery_duration_td

    # --- 4. Trade Statistics ---
    @staticmethod
    def win_rate(portfolio: vbt.Portfolio) -> float:
        """Calculates the Win Rate of trades."""
        if portfolio.trades.count() == 0:
            return np.nan
        stats = portfolio.stats()
        return stats.get('Win Rate [%]', np.nan) / 100.0

    @staticmethod
    def profit_factor(portfolio: vbt.Portfolio) -> float:
        """Calculates the Profit Factor."""
        if portfolio.trades.count() == 0:
            return np.nan
        stats = portfolio.stats()
        return stats.get('Profit Factor', np.nan)

    @staticmethod
    def expectancy(portfolio: vbt.Portfolio) -> float:
        """Calculates the expectancy per trade."""
        if portfolio.trades.count() == 0:
            return np.nan
        stats = portfolio.stats()
        return stats.get('Expectancy', np.nan)

    @staticmethod
    def avg_win(portfolio: vbt.Portfolio) -> float:
        """Calculates the average profit of winning trades."""
        stats = portfolio.stats()
        avg_win_pct = stats.get('Avg Winning Trade [%]', np.nan)
        if np.isnan(avg_win_pct):
            return np.nan
        return (avg_win_pct / 100) * portfolio.init_cash

    @staticmethod
    def avg_loss(portfolio: vbt.Portfolio) -> float:
        """Calculates the average loss of losing trades (as a positive value)."""
        stats = portfolio.stats()
        avg_loss_pct = stats.get('Avg Losing Trade [%]', np.nan)
        if np.isnan(avg_loss_pct):
            return np.nan
        return abs((avg_loss_pct / 100) * portfolio.init_cash)

    @staticmethod
    def largest_win(portfolio: vbt.Portfolio) -> float:
        """Finds the largest single winning trade PnL."""
        stats = portfolio.stats()
        best_trade_pct = stats.get('Best Trade [%]', np.nan)
        if np.isnan(best_trade_pct):
            return np.nan
        return (best_trade_pct / 100) * portfolio.init_cash

    @staticmethod
    def largest_loss(portfolio: vbt.Portfolio) -> float:
        """Finds the largest single losing trade PnL (as a positive value)."""
        stats = portfolio.stats()
        worst_trade_pct = stats.get('Worst Trade [%]', np.nan)
        if np.isnan(worst_trade_pct):
            return np.nan
        return abs((worst_trade_pct / 100) * portfolio.init_cash)

    # --- 5. Time Analysis Metrics ---
    @staticmethod
    def avg_trade_duration(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """Calculates the average duration of trades."""
        if portfolio.trades.count() == 0:
            return None
        stats = portfolio.stats()
        duration = stats.get('Avg Trade Duration', np.nan)
        if pd.isna(duration) or portfolio.wrapper.freq is None:
            return None
        return duration * portfolio.wrapper.freq

    @staticmethod
    def time_in_market(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the percentage of time the strategy was in the market.
        """
        if portfolio.trades.count() == 0:
            return 0.0
        
        try:
            trades_records = portfolio.trades.records_readable
            if trades_records.empty:
                return 0.0
            total_time_in_market = (trades_records['Exit Timestamp'] - trades_records['Entry Timestamp']).sum()
            total_time_in_market_seconds = total_time_in_market.total_seconds()

            start_time = portfolio.wrapper.index[0]
            end_time = portfolio.wrapper.index[-1]
            total_backtest_duration_seconds = (end_time - start_time).total_seconds()
            
            if total_backtest_duration_seconds == 0:
                return np.nan
                
            return total_time_in_market_seconds / total_backtest_duration_seconds
        except Exception as e:
            logger.warning(f"Impossible de calculer 'time_in_market'. Erreur: {e}")
            return np.nan

    @staticmethod
    def longest_trade_duration(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """Finds the duration of the longest trade."""
        if portfolio.trades.count() == 0:
             return None
        trades_records = portfolio.trades.records_readable
        if trades_records.empty:
            return None
        return (trades_records['Exit Timestamp'] - trades_records['Entry Timestamp']).max()

    @staticmethod
    def shortest_trade_duration(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """Finds the duration of the shortest trade."""
        if portfolio.trades.count() == 0:
             return None
        trades_records = portfolio.trades.records_readable
        if trades_records.empty:
            return None
        return (trades_records['Exit Timestamp'] - trades_records['Entry Timestamp']).min()

    # --- 6. Advanced Metrics ---
    @staticmethod
    def kelly_criterion(portfolio: vbt.Portfolio) -> float:
        """Calculates the Kelly Criterion percentage."""
        wr = PerformanceMetrics.win_rate(portfolio)
        avg_w = PerformanceMetrics.avg_win(portfolio)
        avg_l = PerformanceMetrics.avg_loss(portfolio)

        if np.isnan(wr) or np.isnan(avg_w) or np.isnan(avg_l) or avg_l == 0:
            return np.nan

        win_loss_ratio = avg_w / avg_l
        if win_loss_ratio == 0:
            return np.nan

        kelly = wr - ( (1 - wr) / win_loss_ratio )
        return kelly

    @staticmethod
    def risk_of_ruin(
        portfolio: vbt.Portfolio,
        capital_at_risk_pct: float = 0.10,
        ruin_level_pct: float = 0.50
    ) -> float:
        """Estimates the Risk of Ruin (ROR) using a simplified formula."""
        if portfolio.trades.count() < 10:
            return np.nan

        exp = PerformanceMetrics.expectancy(portfolio)
        avg_l_val = PerformanceMetrics.avg_loss(portfolio)

        if np.isnan(exp) or np.isnan(avg_l_val) or avg_l_val == 0:
            return np.nan
        
        initial_capital = portfolio.init_cash
        if initial_capital == 0: return np.nan
        
        edge = exp / avg_l_val

        if edge <= 0:
            return 1.0

        capital_units = (initial_capital * ruin_level_pct) / avg_l_val
        if capital_units <=0: return 1.0

        try:
            trade_returns_pct = portfolio.trades.pnl / portfolio.init_cash
            if len(trade_returns_pct) == 0: return np.nan

            mu_trade = trade_returns_pct.mean()
            sigma_trade = trade_returns_pct.std()

            if sigma_trade == 0 or np.isnan(mu_trade) or np.isnan(sigma_trade): return np.nan
            if mu_trade <=0 : return 1.0
            
            mu_pnl = exp
            sigma_pnl = portfolio.trades.pnl.std()
            if np.isnan(sigma_pnl) or sigma_pnl == 0: return np.nan
            if mu_pnl <=0 : return 1.0
            
            c_ruin_abs = initial_capital * ruin_level_pct
            
            ror_val = np.exp(-2 * mu_pnl * c_ruin_abs / (sigma_pnl**2))
            return np.clip(ror_val, 0, 1)

        except (OverflowError, ValueError, ZeroDivisionError):
            return np.nan
        return np.nan

    @staticmethod
    def monte_carlo_simulation(
        portfolio: vbt.Portfolio,
        n_simulations: int = 1000,
        n_periods_ahead: Optional[int] = None,
        quantile_levels: List[float] = [0.05, 0.50, 0.95]
    ) -> Dict[str, Union[float, pd.Series]]:
        """Performs a Monte Carlo simulation based on trade returns."""
        if portfolio.trades.count() < 1:
            return {"simulated_equity_paths": pd.DataFrame(), "final_equity_quantiles": pd.Series(dtype=float), "median_final_equity": np.nan, "prob_profit": np.nan}

        trade_pnls = portfolio.trades.pnl.values
        if len(trade_pnls) < 1:
             return {"simulated_equity_paths": pd.DataFrame(), "final_equity_quantiles": pd.Series(dtype=float), "median_final_equity": np.nan, "prob_profit": np.nan}

        num_trades_to_sim = n_periods_ahead if n_periods_ahead is not None else portfolio.trades.count()
        initial_capital = portfolio.init_cash

        sim_equity_paths = np.full((num_trades_to_sim, n_simulations), np.nan)
        
        for i in range(n_simulations):
            sim_trade_pnls = np.random.choice(trade_pnls, size=num_trades_to_sim, replace=True)
            current_equity = initial_capital
            path = []
            for pnl in sim_trade_pnls:
                current_equity += pnl
                path.append(current_equity)
            sim_equity_paths[:, i] = path
            
        sim_equity_df = pd.DataFrame(sim_equity_paths, columns=[f'Sim_{i+1}' for i in range(n_simulations)])
        final_equities = sim_equity_df.iloc[-1]
        quantiles = final_equities.quantile(quantile_levels)
        median_final_equity = final_equities.median()
        prob_profit = (final_equities > initial_capital).mean()

        return {"simulated_equity_paths": sim_equity_df, "final_equity_quantiles": quantiles, "median_final_equity": median_final_equity, "prob_profit": prob_profit}

    # --- Aggregation and Export ---
    @staticmethod
    def calculate_all_metrics(
        portfolio: vbt.Portfolio,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        monte_carlo_sims: int = 100
    ) -> Dict[str, Any]:
        """
        Calculates and aggregates all defined performance metrics.
        """
        if not isinstance(portfolio, vbt.Portfolio):
            raise TypeError("Input `portfolio` must be a vectorbt.Portfolio object.")

        metrics = {}

        try:
            vbt_stats = portfolio.stats()
            for k, v in vbt_stats.items():
                if isinstance(v, pd.Series):
                    if not v.empty:
                        metrics[f"vbt_{k.replace(' ', '_').lower()}"] = v.iloc[0]
                    else:
                        metrics[f"vbt_{k.replace(' ', '_').lower()}"] = np.nan
                else:
                    metrics[f"vbt_{k.replace(' ', '_').lower()}"] = v
        except Exception as e:
            pass

        metrics['returns'] = {
            "total_return_pct": PerformanceMetrics.total_return(portfolio) * 100,
            "annualized_return_pct": PerformanceMetrics.annualized_return(portfolio) * 100,
        }

        metrics['risk_ratios'] = {
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(portfolio, risk_free_rate_annual),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(portfolio, risk_free_rate_annual),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(portfolio),
            "information_ratio": PerformanceMetrics.information_ratio(portfolio, benchmark_returns) if benchmark_returns is not None else np.nan,
        }

        mdd_duration = PerformanceMetrics.max_drawdown_duration(portfolio)
        recovery_t = PerformanceMetrics.recovery_time(portfolio)
        metrics['drawdown'] = {
            "max_drawdown_pct": PerformanceMetrics.max_drawdown(portfolio) * 100,
            "max_drawdown_duration_days": mdd_duration.days if isinstance(mdd_duration, pd.Timedelta) else mdd_duration,
            "average_recovery_time_days": recovery_t.days if recovery_t else np.nan,
        }

        metrics['trade_stats'] = {
            "total_trades": portfolio.trades.count(),
            "win_rate_pct": PerformanceMetrics.win_rate(portfolio) * 100 if portfolio.trades.count() > 0 else np.nan,
            "profit_factor": PerformanceMetrics.profit_factor(portfolio),
            "expectancy": PerformanceMetrics.expectancy(portfolio),
            "avg_win_pnl": PerformanceMetrics.avg_win(portfolio),
            "avg_loss_pnl": PerformanceMetrics.avg_loss(portfolio),
            "largest_win_pnl": PerformanceMetrics.largest_win(portfolio),
            "largest_loss_pnl": PerformanceMetrics.largest_loss(portfolio),
        }
        
        avg_trade_dur = PerformanceMetrics.avg_trade_duration(portfolio)
        long_trade_dur = PerformanceMetrics.longest_trade_duration(portfolio)
        short_trade_dur = PerformanceMetrics.shortest_trade_duration(portfolio)

        metrics['time_analysis'] = {
            "avg_trade_duration_str": str(avg_trade_dur) if avg_trade_dur else "N/A",
            "time_in_market_pct": PerformanceMetrics.time_in_market(portfolio) * 100,
            "longest_trade_duration_str": str(long_trade_dur) if long_trade_dur else "N/A",
            "shortest_trade_duration_str": str(short_trade_dur) if short_trade_dur else "N/A",
        }

        metrics['advanced'] = {
            "kelly_criterion_pct": PerformanceMetrics.kelly_criterion(portfolio) * 100,
            "risk_of_ruin_pct": PerformanceMetrics.risk_of_ruin(portfolio) * 100,
        }

        if monte_carlo_sims > 0:
            mc_results = PerformanceMetrics.monte_carlo_simulation(portfolio, n_simulations=monte_carlo_sims)
            metrics['monte_carlo'] = {
                "median_final_equity": mc_results["median_final_equity"],
                "prob_profit_pct": mc_results["prob_profit"] * 100 if not np.isnan(mc_results["prob_profit"]) else np.nan,
                "final_equity_5th_percentile": mc_results["final_equity_quantiles"].get(0.05, np.nan),
                "final_equity_95th_percentile": mc_results["final_equity_quantiles"].get(0.95, np.nan),
            }
        
        flat_metrics = {}
        for category, cat_metrics in metrics.items():
            if isinstance(cat_metrics, dict):
                for key, value in cat_metrics.items():
                    flat_metrics[f"{category}_{key}"] = value
            else:
                flat_metrics[category] = cat_metrics
        
        flat_metrics['general_start_date'] = portfolio.wrapper.index[0]
        flat_metrics['general_end_date'] = portfolio.wrapper.index[-1]
        flat_metrics['general_duration_days'] = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
        flat_metrics['general_initial_capital'] = portfolio.init_cash
        flat_metrics['general_final_value'] = portfolio.value().iloc[-1]
        # --- CORRECTION: Ajout de la métrique PnL nette totale ---
        flat_metrics['general_total_net_pnl'] = portfolio.value().iloc[-1] - portfolio.init_cash
        # --- FIN DE LA CORRECTION ---
        
        return flat_metrics

    @staticmethod
    def metrics_to_dataframe(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts the aggregated metrics dictionary to a pandas DataFrame.
        """
        simple_metrics = {k: v for k, v in metrics_dict.items() 
                          if isinstance(v, (int, float, str, bool, pd.Timestamp, pd.Timedelta)) or pd.isna(v)}
        
        try:
            df = pd.DataFrame.from_dict(simple_metrics, orient='index', columns=['Value'])
            df.index.name = "Metric"
            return df
        except Exception as e:
            return pd.DataFrame(list(simple_metrics.items()), columns=['Metric', 'Value']).set_index('Metric')

    # --- Rolling Metrics (Examples using vectorbt's capabilities) ---
    @staticmethod
    def rolling_sharpe_ratio(
        portfolio: vbt.Portfolio,
        window: int,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs
    ) -> pd.Series:
        """Calculates rolling Sharpe Ratio."""
        return portfolio.rolling_sharpe_ratio(window, risk_free=risk_free_rate_annual, **kwargs)

    @staticmethod
    def rolling_sortino_ratio(
        portfolio: vbt.Portfolio,
        window: int,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs
    ) -> pd.Series:
        """Calculates rolling Sortino Ratio."""
        return portfolio.rolling_sortino_ratio(window, required_return=risk_free_rate_annual, **kwargs)
