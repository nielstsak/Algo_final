# src/backtesting/performance_metrics.py
import pandas as pd
import numpy as np
import vectorbt as vbt
from scipy.stats import norm
from typing import Dict, List, Optional, Any, Tuple, Union

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
        else: # Fallback to daily for other frequencies, or assume returns are already at the correct frequency
            # This simplification might need adjustment based on the actual freq of returns
            days_in_period = returns_period.total_seconds() / (24 * 3600)
            if days_in_period > 0 and TRADING_DAYS_PER_YEAR / days_in_period > 0:
                return (1 + risk_free_rate_annual)**(days_in_period / TRADING_DAYS_PER_YEAR) - 1
            return 0.0 # Default if period cannot be determined


    # --- 1. Return Metrics ---
    @staticmethod
    def total_return(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the total cumulative return of the strategy.

        Formula:
        $$ R_{total} = \\frac{Value_{final} - Value_{initial}}{Value_{initial}} $$
        Or directly from vectorbt: `portfolio.total_return()`

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The total return as a float.
        """
        return portfolio.total_return()

    @staticmethod
    def annualized_return(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the annualized return.

        Formula (simplified):
        $$ R_{annualized} = (1 + R_{total})^{\\frac{N_{periods\_in\_year}}{N_{total\_periods}}} - 1 $$
        Or directly from vectorbt: `portfolio.annualized_return()`

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The annualized return as a float.
        """
        return portfolio.annualized_return()

    @staticmethod
    def daily_returns(portfolio: vbt.Portfolio) -> pd.Series:
        """
        Extracts daily returns. If portfolio frequency is not daily, it resamples.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            A pandas Series of daily returns.
        """
        if portfolio.wrapper.freq == pd.Timedelta(days=1):
            return portfolio.returns()
        else:
            # Resample portfolio value to daily, then calculate returns
            # This is an approximation if trades are not daily.
            # For true daily returns, the input data to portfolio should be daily.
            daily_values = portfolio.value().resample('D').last().ffill()
            return daily_values.pct_change().dropna()


    @staticmethod
    def monthly_returns(portfolio: vbt.Portfolio) -> pd.Series:
        """
        Calculates monthly returns from portfolio values.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            A pandas Series of monthly returns.
        """
        monthly_values = portfolio.value().resample('M').last()
        return monthly_values.pct_change().dropna()

    # --- 2. Risk Ratios ---
    @staticmethod
    def sharpe_ratio(
        portfolio: vbt.Portfolio,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs # For vbt.Portfolio.sharpe_ratio options
    ) -> float:
        """
        Calculates the Sharpe Ratio.

        Formula:
        $$ S = \\frac{E[R_p - R_f]}{\\sigma_p} $$
        Where $E[R_p]$ is the expected portfolio return, $R_f$ is the risk-free rate,
        and $\\sigma_p$ is the standard deviation of the portfolio's excess return.
        vectorbt calculates this based on the portfolio's return frequency.

        Args:
            portfolio: A vectorbt.Portfolio object.
            risk_free_rate_annual: Annual risk-free rate.
            **kwargs: Additional arguments for `portfolio.sharpe_ratio()`.

        Returns:
            The Sharpe Ratio as a float.
        """
        return portfolio.sharpe_ratio(risk_free_rate=risk_free_rate_annual, **kwargs)

    @staticmethod
    def sortino_ratio(
        portfolio: vbt.Portfolio,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs # For vbt.Portfolio.sortino_ratio options
    ) -> float:
        """
        Calculates the Sortino Ratio.

        Formula:
        $$ Sortino = \\frac{E[R_p - R_f]}{\\sigma_d} $$
        Where $\\sigma_d$ is the standard deviation of downside returns.

        Args:
            portfolio: A vectorbt.Portfolio object.
            risk_free_rate_annual: Annual risk-free rate.
            **kwargs: Additional arguments for `portfolio.sortino_ratio()`.

        Returns:
            The Sortino Ratio as a float.
        """
        return portfolio.sortino_ratio(risk_free_rate=risk_free_rate_annual, **kwargs)

    @staticmethod
    def calmar_ratio(portfolio: vbt.Portfolio, **kwargs) -> float:
        """
        Calculates the Calmar Ratio.

        Formula:
        $$ Calmar = \\frac{R_{annualized}}{|MDD|} $$
        Where $MDD$ is the Maximum Drawdown.

        Args:
            portfolio: A vectorbt.Portfolio object.
            **kwargs: Additional arguments for `portfolio.calmar_ratio()`.

        Returns:
            The Calmar Ratio as a float.
        """
        return portfolio.calmar_ratio(**kwargs)

    @staticmethod
    def information_ratio(
        portfolio: vbt.Portfolio,
        benchmark_returns: pd.Series,
        **kwargs # For vbt.Portfolio.information_ratio options
    ) -> float:
        """
        Calculates the Information Ratio.

        Formula:
        $$ IR = \\frac{E[R_p - R_b]}{\\sigma(R_p - R_b)} $$
        Where $R_b$ is the benchmark return and $\\sigma(R_p - R_b)$ is the standard
        deviation of the active return (portfolio return - benchmark return).

        Args:
            portfolio: A vectorbt.Portfolio object.
            benchmark_returns: A pandas Series of benchmark returns, aligned with portfolio.
            **kwargs: Additional arguments for `portfolio.information_ratio()`.

        Returns:
            The Information Ratio as a float.
        """
        if benchmark_returns is None or benchmark_returns.empty:
            return np.nan
        try:
            # Ensure benchmark_returns is aligned with portfolio.returns()
            # This might involve reindexing or intersection if not already aligned.
            # For simplicity, vectorbt's information_ratio handles this if benchmark_returns is passed.
            return portfolio.information_ratio(benchmark_returns, **kwargs)
        except Exception as e:
            # vectorbt might raise error if benchmark is not suitable
            # logger.warning(f"Could not calculate Information Ratio: {e}")
            return np.nan


    # --- 3. Drawdown Metrics ---
    @staticmethod
    def max_drawdown(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the Maximum Drawdown (MDD).
        MDD is the largest peak-to-trough decline during a specific period.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The Maximum Drawdown as a positive float (e.g., 0.2 for 20% drawdown).
        """
        return portfolio.max_drawdown()

    @staticmethod
    def max_drawdown_duration(portfolio: vbt.Portfolio) -> Union[pd.Timedelta, float]:
        """
        Calculates the duration of the longest drawdown period.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The duration as a pandas.Timedelta or float (in units of frequency).
        """
        return portfolio.max_drawdown_duration()

    @staticmethod
    def underwater_curve(portfolio: vbt.Portfolio) -> pd.Series:
        """
        Returns the underwater curve (drawdowns over time).

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            A pandas Series representing the drawdown at each point in time.
        """
        return portfolio.drawdown()

    @staticmethod
    def recovery_time(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """
        Calculates the average or longest time to recover from drawdowns.
        This is a simplified version focusing on the recovery from the max drawdown.
        A more comprehensive analysis would look at all drawdowns.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The time to recover from the maximum drawdown, or None if not recovered.
        """
        drawdowns = portfolio.drawdowns # Access the DrawdownsAccessor
        if drawdowns.records_arr.shape[0] == 0:
            return None

        # Find the maximum drawdown record
        mdd_record = drawdowns.records_arr[np.argmax(drawdowns.records_arr['Drawdown'])]

        if pd.isna(mdd_record['Recovery Date']): # Not recovered
            return None

        recovery_duration_td = pd.to_datetime(mdd_record['Recovery Date']) - pd.to_datetime(mdd_record['Valley Date'])
        return recovery_duration_td


    # --- 4. Trade Statistics ---
    # These methods often rely on portfolio.trades accessor
    @staticmethod
    def win_rate(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the Win Rate of trades.

        Formula:
        $$ WinRate = \\frac{Number \\ of \\ Winning \\ Trades}{Total \\ Number \\ of \\ Trades} $$

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The Win Rate as a float (e.g., 0.6 for 60%). Returns np.nan if no trades.
        """
        if portfolio.trades.count() == 0:
            return np.nan
        return portfolio.trades.win_rate()

    @staticmethod
    def profit_factor(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the Profit Factor.

        Formula:
        $$ ProfitFactor = \\frac{Gross \\ Profit}{ |Gross \\ Loss|} $$

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The Profit Factor. Returns np.nan if no losses or no trades.
        """
        if portfolio.trades.count() == 0:
            return np.nan
        gross_profit = portfolio.trades.winning_trades().pnl.sum()
        gross_loss = portfolio.trades.losing_trades().pnl.sum()
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else np.nan # Or define as gross_profit if desired
        return gross_profit / abs(gross_loss)


    @staticmethod
    def expectancy(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the expectancy per trade.

        Formula:
        $$ Expectancy = (WinRate \\times AvgWin) - (LossRate \\times AvgLoss) $$
        Note: AvgLoss is typically a negative value, so it's $|AvgLoss|$ or handled by vbt.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The expectancy per trade. Returns np.nan if no trades.
        """
        if portfolio.trades.count() == 0:
            return np.nan
        return portfolio.trades.expectancy()


    @staticmethod
    def avg_win(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the average profit of winning trades.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Average PnL of winning trades. Returns np.nan if no winning trades.
        """
        if portfolio.trades.winning_trades().count() == 0:
            return np.nan
        return portfolio.trades.avg_winning_trade()

    @staticmethod
    def avg_loss(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the average loss of losing trades (as a positive value).

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Average PnL of losing trades (absolute value). Np.nan if no losing trades.
        """
        if portfolio.trades.losing_trades().count() == 0:
            return np.nan
        # vbt.avg_losing_trade returns negative, so abs()
        return abs(portfolio.trades.avg_losing_trade())


    @staticmethod
    def largest_win(portfolio: vbt.Portfolio) -> float:
        """
        Finds the largest single winning trade PnL.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The PnL of the largest winning trade. Returns np.nan if no winning trades.
        """
        winning_trades_pnl = portfolio.trades.winning_trades().pnl
        if winning_trades_pnl.empty:
            return np.nan
        return winning_trades_pnl.max()

    @staticmethod
    def largest_loss(portfolio: vbt.Portfolio) -> float:
        """
        Finds the largest single losing trade PnL (as a positive value).

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            The absolute PnL of the largest losing trade. Np.nan if no losing trades.
        """
        losing_trades_pnl = portfolio.trades.losing_trades().pnl
        if losing_trades_pnl.empty:
            return np.nan
        return abs(losing_trades_pnl.min()) # min will be most negative, abs makes it positive

    # --- 5. Time Analysis Metrics ---
    @staticmethod
    def avg_trade_duration(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """
        Calculates the average duration of trades.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Average trade duration as pd.Timedelta or None if no trades.
        """
        if portfolio.trades.count() == 0:
            return None
        # VectorBT avg_trade_duration returns duration in number of bars. Convert to Timedelta.
        avg_duration_bars = portfolio.trades.avg_trade_duration()
        if np.isnan(avg_duration_bars) or portfolio.wrapper.freq is None:
            return None
        return avg_duration_bars * portfolio.wrapper.freq

    @staticmethod
    def time_in_market(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the percentage of time the strategy was in the market.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Percentage of time in market (0.0 to 1.0). Np.nan if no trades or total time is zero.
        """
        if portfolio.trades.count() == 0:
            return np.nan

        total_time_in_market_bars = portfolio.trades.total_duration() # Total duration in bars
        if np.isnan(total_time_in_market_bars) or portfolio.wrapper.freq is None:
            return np.nan

        total_time_in_market = total_time_in_market_bars * portfolio.wrapper.freq
        
        start_time = portfolio.wrapper.index[0]
        end_time = portfolio.wrapper.index[-1]
        total_backtest_duration = end_time - start_time
        
        if total_backtest_duration.total_seconds() == 0:
            return np.nan
            
        return total_time_in_market.total_seconds() / total_backtest_duration.total_seconds()


    @staticmethod
    def longest_trade_duration(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """
        Finds the duration of the longest trade.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Duration of the longest trade as pd.Timedelta or None if no trades.
        """
        trades_records = portfolio.trades.records
        if trades_records.empty:
            return None
        longest_duration_bars = (trades_records['Exit Timestamp'] - trades_records['Entry Timestamp']).max()
        if portfolio.wrapper.freq is None or pd.isna(longest_duration_bars):
             return None
        return longest_duration_bars * portfolio.wrapper.freq


    @staticmethod
    def shortest_trade_duration(portfolio: vbt.Portfolio) -> Optional[pd.Timedelta]:
        """
        Finds the duration of the shortest trade.

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Duration of the shortest trade as pd.Timedelta or None if no trades.
        """
        trades_records = portfolio.trades.records
        if trades_records.empty:
            return None
        shortest_duration_bars = (trades_records['Exit Timestamp'] - trades_records['Entry Timestamp']).min()
        if portfolio.wrapper.freq is None or pd.isna(shortest_duration_bars):
             return None
        return shortest_duration_bars * portfolio.wrapper.freq


    # --- 6. Advanced Metrics ---
    @staticmethod
    def kelly_criterion(portfolio: vbt.Portfolio) -> float:
        """
        Calculates the Kelly Criterion percentage.
        Assumes PnL of trades are relative to the capital at the time of trade,
        or uses average win/loss percentages. For simplicity, we use win rate and profit_factor.

        Formula (simplified using average win/loss amounts):
        $$ K\% = W - \\frac{1 - W}{R} $$
        Where $W$ is the probability of a win (WinRate).
        $R$ is the win/loss ratio (AvgWin / AvgLoss).

        Args:
            portfolio: A vectorbt.Portfolio object.

        Returns:
            Kelly Criterion percentage. Np.nan if unable to calculate.
        """
        wr = PerformanceMetrics.win_rate(portfolio)
        avg_w = PerformanceMetrics.avg_win(portfolio)
        avg_l = PerformanceMetrics.avg_loss(portfolio) # Already positive

        if np.isnan(wr) or np.isnan(avg_w) or np.isnan(avg_l) or avg_l == 0:
            return np.nan

        win_loss_ratio = avg_w / avg_l
        if win_loss_ratio == 0: # Should not happen if avg_w is not nan and > 0
            return np.nan

        kelly = wr - ( (1 - wr) / win_loss_ratio )
        return kelly

    @staticmethod
    def risk_of_ruin(
        portfolio: vbt.Portfolio,
        capital_at_risk_pct: float = 0.10, # Max percentage of capital risked per trade
        ruin_level_pct: float = 0.50 # Percentage of capital loss defined as ruin
    ) -> float:
        """
        Estimates the Risk of Ruin (ROR) using a simplified formula.
        This is a complex topic, and this is a basic estimation.

        Formula (approximate, based on Perry Kaufman's adaptation of Thorp's formula):
        Relies on the 'trading edge' or expectancy.
        $$ ROR = \\left( \\frac{1 - Edge}{1 + Edge} \\right)^{CapitalUnits} $$
        Where $Edge = Expectancy / AvgLoss$ (if expectancy and avg loss are in same units).
        $CapitalUnits = TotalCapital / CapitalRiskedPerTrade$.
        Or, more directly: $Edge = (W \\times (R+1) - 1) / R$, where $W$ is win rate, $R$ is win/loss ratio.
        $CapitalUnits = \\frac{ln(1 - RuinLevel)}{ln(1 - CapitalAtRiskPerTrade)}$.

        A simpler approximation for ROR:
        If average loss percentage $L_{pct}$ and win rate $W$:
        $ROR = ((1 - W) / W)^{RUIN\_LEVEL / L_{pct}}$ if using percentage losses.

        This implementation uses portfolio's expectancy and average loss values.

        Args:
            portfolio: A vectorbt.Portfolio object.
            capital_at_risk_pct: The maximum percentage of capital one is willing to lose on a single trade.
                                 This is hard to derive directly from vbt portfolio for a generic formula.
                                 This implementation will use average loss as a proxy for capital at risk.
            ruin_level_pct: The percentage of total capital loss that constitutes "ruin".

        Returns:
            Estimated Risk of Ruin (0.0 to 1.0). Np.nan if unable to calculate.
        """
        if portfolio.trades.count() < 10: # Needs sufficient trades
            return np.nan

        exp = PerformanceMetrics.expectancy(portfolio) # PnL per trade
        avg_l_val = PerformanceMetrics.avg_loss(portfolio) # Absolute PnL

        if np.isnan(exp) or np.isnan(avg_l_val) or avg_l_val == 0:
            return np.nan
        
        initial_capital = portfolio.init_cash
        if initial_capital == 0: return np.nan

        # Approximation: Edge as ratio of expectancy to average loss
        # This edge is not necessarily between -1 and 1.
        # If expectancy is positive, edge > 0. If negative, edge < 0.
        edge = exp / avg_l_val # How many average losses does one expectancy unit represent?
                               # If exp > 0, edge > 0.

        if edge <= 0: # Negative or zero expectancy implies high risk of ruin
            return 1.0

        # Capital Units: How many average losses can the 'ruin level' sustain?
        # CapitalAtRisk = avg_l_val (as a proxy)
        # TotalCapitalToLoseBeforeRuin = initial_capital * ruin_level_pct
        # CapitalUnits = TotalCapitalToLoseBeforeRuin / CapitalAtRisk
        capital_units = (initial_capital * ruin_level_pct) / avg_l_val
        if capital_units <=0: return 1.0

        try:
            # Using formula: ROR = ((1 - Edge) / (1 + Edge))^CapitalUnits
            # This formula assumes Edge is a probability (0 to 1 for positive expectancy),
            # which our current `edge` is not. Let's use a more robust form if possible
            # or acknowledge this is a very rough estimate.

            # A more common formula: ROR = exp( -2 * Edge * CapitalUnits )
            # This assumes Edge = (W * R - (1-W)) / R where R is win/loss ratio
            # Let's try another common one: ROR = ((LosingProb / WinningProb)^CapitalUnitsAtRisk)
            # Or ROR = exp(-2 * Z * C) where Z = (W*AvgW - L*AvgL)/std_dev_of_pnl, C = capital_units_at_risk
            # This is getting too complex for a generic function without more assumptions.

            # For now, stick to a very simplified approach:
            # If we have a positive edge, the risk of ruin decreases as capital units increase.
            # Use formula ROR = ((1-A)/(A)) ^ C (where A is probability of winning an amount equal to what's risked)
            # This is difficult to map directly.

            # Alternative simple formula: if p = win_rate, q = 1-p, R = avg_win/avg_loss
            # ROR = ( (q/p)^R ) ^ (CapitalUnits) -- this also seems off.

            # Let's use the one involving `edge = exp / avg_l_val` and capital_units
            # ROR = exp( -2 * edge_norm * capital_units), where edge_norm is a normalized edge
            # This is still a simplification.
            # Acknowledging high approximation:
            if edge > 0 and capital_units > 0:
                # This specific formula is less standard. A placeholder for a more rigorous one.
                # For a positive edge, ruin should decrease.
                # Let's use a more common one based on geometric Brownian motion approximation:
                # ROR = exp(-2 * mu_trade * C_ruin / sigma_trade^2)
                # where mu_trade is avg trade return, C_ruin is capital level for ruin, sigma_trade is std of trade returns
                
                trade_returns_pct = portfolio.trades.pnl / portfolio.init_cash # Very rough if trades are not on full capital
                if trade_returns_pct.empty: return np.nan

                mu_trade = trade_returns_pct.mean()
                sigma_trade = trade_returns_pct.std()

                if sigma_trade == 0 or np.isnan(mu_trade) or np.isnan(sigma_trade): return np.nan
                if mu_trade <=0 : return 1.0 # Non-positive average trade return means ruin is likely

                # C_ruin: if current capital is C0, ruin is at C0 * (1-ruin_level_pct)
                # The "distance" to ruin is C0 * ruin_level_pct
                # The formula needs absolute capital level for ruin relative to initial capital.
                # Let's use the number of average losses that lead to ruin.
                # C = (InitialCapital * RuinLevelPct) / AvgLossAmount
                # If mu_trade is positive, then ROR = exp(-2 * (mu_trade / sigma_trade^2) * (InitialCapital*RuinLevelPct))
                # This means mu_trade should be in currency units if InitialCapital*RuinLevelPct is.
                # So, use mu_pnl = exp, sigma_pnl = portfolio.trades.pnl.std()
                mu_pnl = exp
                sigma_pnl = portfolio.trades.pnl.std()
                if np.isnan(sigma_pnl) or sigma_pnl == 0: return np.nan
                if mu_pnl <=0 : return 1.0
                
                c_ruin_abs = initial_capital * ruin_level_pct # Amount of capital that can be lost
                
                ror_val = np.exp(-2 * mu_pnl * c_ruin_abs / (sigma_pnl**2))
                return np.clip(ror_val, 0, 1)

        except (OverflowError, ValueError, ZeroDivisionError):
            return np.nan # Calculation failed
        return np.nan # Default if logic paths don't compute

    @staticmethod
    def monte_carlo_simulation(
        portfolio: vbt.Portfolio,
        n_simulations: int = 1000,
        n_periods_ahead: Optional[int] = None, # If None, use length of original backtest
        quantile_levels: List[float] = [0.05, 0.50, 0.95]
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Performs a Monte Carlo simulation based on trade returns.

        Args:
            portfolio: A vectorbt.Portfolio object.
            n_simulations: Number of simulation paths to generate.
            n_periods_ahead: Number of periods (trades) to simulate ahead.
                             If None, uses the number of trades in the portfolio.
            quantile_levels: Quantile levels for the final equity distribution.

        Returns:
            A dictionary containing:
                - 'simulated_equity_paths': DataFrame of simulated equity paths.
                - 'final_equity_quantiles': Series of final equity values at specified quantiles.
                - 'median_final_equity': Median final equity.
                - 'prob_profit': Probability of ending with profit.
        """
        if portfolio.trades.count() < 1:
            return {
                "simulated_equity_paths": pd.DataFrame(),
                "final_equity_quantiles": pd.Series(dtype=float),
                "median_final_equity": np.nan,
                "prob_profit": np.nan
            }

        trade_pnls = portfolio.trades.pnl.to_numpy()
        if len(trade_pnls) < 1: # Not enough unique trade PnLs
             return {
                "simulated_equity_paths": pd.DataFrame(),
                "final_equity_quantiles": pd.Series(dtype=float),
                "median_final_equity": np.nan,
                "prob_profit": np.nan
            }


        num_trades_to_sim = n_periods_ahead if n_periods_ahead is not None else portfolio.trades.count()
        initial_capital = portfolio.init_cash

        sim_equity_paths = np.full((num_trades_to_sim, n_simulations), np.nan)
        
        for i in range(n_simulations):
            # Sample with replacement from actual trade P&Ls
            sim_trade_pnls = np.random.choice(trade_pnls, size=num_trades_to_sim, replace=True)
            current_equity = initial_capital
            path = []
            for pnl in sim_trade_pnls:
                current_equity += pnl
                path.append(current_equity)
            sim_equity_paths[:, i] = path
            
        sim_equity_df = pd.DataFrame(sim_equity_paths, 
                                     columns=[f'Sim_{i+1}' for i in range(n_simulations)])

        final_equities = sim_equity_df.iloc[-1]
        
        quantiles = final_equities.quantile(quantile_levels)
        median_final_equity = final_equities.median()
        prob_profit = (final_equities > initial_capital).mean()

        return {
            "simulated_equity_paths": sim_equity_df,
            "final_equity_quantiles": quantiles,
            "median_final_equity": median_final_equity,
            "prob_profit": prob_profit
        }

    # --- Aggregation and Export ---
    @staticmethod
    def calculate_all_metrics(
        portfolio: vbt.Portfolio,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        monte_carlo_sims: int = 100 # Set to 0 to disable MC
    ) -> Dict[str, Any]:
        """
        Calculates and aggregates all defined performance metrics.

        Args:
            portfolio: A vectorbt.Portfolio object.
            benchmark_returns: Optional pandas Series of benchmark returns.
            risk_free_rate_annual: Annual risk-free rate.
            monte_carlo_sims: Number of simulations for Monte Carlo.

        Returns:
            A dictionary with all calculated metrics, organized by category.
        """
        if not isinstance(portfolio, vbt.Portfolio):
            raise TypeError("Input `portfolio` must be a vectorbt.Portfolio object.")

        metrics = {}

        # Basic Portfolio Stats from vectorbt (can be used as a base)
        try:
            vbt_stats = portfolio.stats(settings=dict(risk_free_rate=risk_free_rate_annual))
            # Filter out any Series objects from vbt_stats if we want single values
            for k, v in vbt_stats.items():
                if isinstance(v, pd.Series):
                    # Take the first value if it's a Series (e.g. from multiple columns)
                    # This assumes the portfolio is for a single strategy run.
                    if not v.empty:
                        metrics[f"vbt_{k.replace(' ', '_').lower()}"] = v.iloc[0]
                    else:
                        metrics[f"vbt_{k.replace(' ', '_').lower()}"] = np.nan
                else:
                     metrics[f"vbt_{k.replace(' ', '_').lower()}"] = v

        except Exception as e:
            # logger.warning(f"Could not get base vectorbt stats: {e}")
            # Initialize with some known keys from vbt.Portfolio.stats_defaults
            # to avoid KeyErrors later if vbt_stats failed partially.
            # This is a fallback; ideally, portfolio.stats() should work.
            pass


        # Return Metrics
        metrics['returns'] = {
            "total_return_pct": PerformanceMetrics.total_return(portfolio) * 100,
            "annualized_return_pct": PerformanceMetrics.annualized_return(portfolio) * 100,
            # "daily_returns_avg_pct": PerformanceMetrics.daily_returns(portfolio).mean() * 100 if not PerformanceMetrics.daily_returns(portfolio).empty else np.nan,
            # "monthly_returns_avg_pct": PerformanceMetrics.monthly_returns(portfolio).mean() * 100 if not PerformanceMetrics.monthly_returns(portfolio).empty else np.nan,
        }

        # Risk Ratios
        metrics['risk_ratios'] = {
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(portfolio, risk_free_rate_annual),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(portfolio, risk_free_rate_annual),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(portfolio),
            "information_ratio": PerformanceMetrics.information_ratio(portfolio, benchmark_returns) if benchmark_returns is not None else np.nan,
        }

        # Drawdown Metrics
        mdd_duration = PerformanceMetrics.max_drawdown_duration(portfolio)
        recovery_t = PerformanceMetrics.recovery_time(portfolio)
        metrics['drawdown'] = {
            "max_drawdown_pct": PerformanceMetrics.max_drawdown(portfolio) * 100,
            "max_drawdown_duration_days": mdd_duration.days if isinstance(mdd_duration, pd.Timedelta) else mdd_duration, # Assuming freq translates to days if not Timedelta
            "average_recovery_time_days": recovery_t.days if recovery_t else np.nan,
        }

        # Trade Statistics
        metrics['trade_stats'] = {
            "total_trades": portfolio.trades.count(),
            "win_rate_pct": PerformanceMetrics.win_rate(portfolio) * 100 if portfolio.trades.count() > 0 else np.nan,
            "profit_factor": PerformanceMetrics.profit_factor(portfolio),
            "expectancy": PerformanceMetrics.expectancy(portfolio),
            "avg_win_pnl": PerformanceMetrics.avg_win(portfolio),
            "avg_loss_pnl": PerformanceMetrics.avg_loss(portfolio), # Already positive
            "largest_win_pnl": PerformanceMetrics.largest_win(portfolio),
            "largest_loss_pnl": PerformanceMetrics.largest_loss(portfolio), # Already positive
        }
        
        # Time Analysis
        avg_trade_dur = PerformanceMetrics.avg_trade_duration(portfolio)
        long_trade_dur = PerformanceMetrics.longest_trade_duration(portfolio)
        short_trade_dur = PerformanceMetrics.shortest_trade_duration(portfolio)

        metrics['time_analysis'] = {
            "avg_trade_duration_str": str(avg_trade_dur) if avg_trade_dur else "N/A",
            "time_in_market_pct": PerformanceMetrics.time_in_market(portfolio) * 100,
            "longest_trade_duration_str": str(long_trade_dur) if long_trade_dur else "N/A",
            "shortest_trade_duration_str": str(short_trade_dur) if short_trade_dur else "N/A",
        }

        # Advanced Metrics
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
        
        # Flatten the dictionary for easier DataFrame conversion or direct use
        flat_metrics = {}
        for category, cat_metrics in metrics.items():
            if isinstance(cat_metrics, dict):
                 for key, value in cat_metrics.items():
                    flat_metrics[f"{category}_{key}"] = value
            else: # For direct vbt_stats that were not categorized
                flat_metrics[category] = cat_metrics
        
        # Add some general portfolio info
        flat_metrics['general_start_date'] = portfolio.wrapper.index[0]
        flat_metrics['general_end_date'] = portfolio.wrapper.index[-1]
        flat_metrics['general_duration_days'] = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days
        flat_metrics['general_initial_capital'] = portfolio.init_cash
        flat_metrics['general_final_value'] = portfolio.value()[-1]
        
        return flat_metrics

    @staticmethod
    def metrics_to_dataframe(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts the aggregated metrics dictionary to a pandas DataFrame.

        Args:
            metrics_dict: A dictionary of metrics, typically from calculate_all_metrics.

        Returns:
            A pandas DataFrame with metrics.
        """
        # Filter out complex objects if any were accidentally included for the summary DataFrame
        simple_metrics = {k: v for k, v in metrics_dict.items() 
                          if isinstance(v, (int, float, str, bool, pd.Timestamp, pd.Timedelta)) or pd.isna(v)}
        
        try:
            df = pd.DataFrame.from_dict(simple_metrics, orient='index', columns=['Value'])
            df.index.name = "Metric"
            return df
        except Exception as e:
            # logger.error(f"Could not convert metrics to DataFrame: {e}")
            # Fallback to a simpler DataFrame creation
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
        return portfolio.rolling_sharpe_ratio(window, risk_free_rate=risk_free_rate_annual, **kwargs)

    @staticmethod
    def rolling_sortino_ratio(
        portfolio: vbt.Portfolio,
        window: int,
        risk_free_rate_annual: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        **kwargs
    ) -> pd.Series:
        """Calculates rolling Sortino Ratio."""
        return portfolio.rolling_sortino_ratio(window, risk_free_rate=risk_free_rate_annual, **kwargs)

# Example Usage (comment out when used as a module):
# if __name__ == "__main__":
#     # This requires a portfolio object.
#     # You would typically get this from running a backtest with VectorBTEngine
#     # For demonstration, let's try to create a mock or simple portfolio if possible
#     # or state that this part needs a real portfolio.
#     print("PerformanceMetrics class defined. To use, obtain a vectorbt.Portfolio object from a backtest.")
#
#     # Example: (requires data and signals)
#     # price = pd.Series(np.random.rand(1000) + 100, index=pd.date_range("2020-01-01", periods=1000))
#     # entries = pd.Series(False, index=price.index)
#     # entries.iloc[::20] = True
#     # exits = pd.Series(False, index=price.index)
#     # exits.iloc[10::20] = True
#     # pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10000)
#
#     # if pf.trades.count() > 0 :
#     #     all_mets = PerformanceMetrics.calculate_all_metrics(pf, monte_carlo_sims=100)
#     #     print("\nAll Metrics (Dictionary):")
#     #     for k,v in all_mets.items():
#     #         print(f"  {k}: {v}")
#
#     #     mets_df = PerformanceMetrics.metrics_to_dataframe(all_mets)
#     #     print("\nMetrics DataFrame:")
#     #     print(mets_df)
#
#     #     rolling_sharpe = PerformanceMetrics.rolling_sharpe_ratio(pf, window=100)
#     #     print("\nRolling Sharpe (100-period):")
#     #     print(rolling_sharpe.tail())
#     # else:
#     #     print("Mock portfolio has no trades, cannot calculate metrics.")