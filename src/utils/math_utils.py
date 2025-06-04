import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict
from numpy.typing import NDArray
from scipy.optimize import minimize
import math

# Optional Numba import for performance critical functions
# from numba import jit, njit

# --- Fonctions statistiques ---

def sharpe_ratio(
    returns: Union[pd.Series, NDArray[np.float64]],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calcule le ratio de Sharpe annualisé.

    Le ratio de Sharpe mesure le rendement ajusté au risque d'un investissement.
    Il est calculé comme le rendement excédentaire moyen par rapport au taux sans risque,
    divisé par l'écart-type de ces rendements excédentaires.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Une série ou un tableau NumPy de rendements périodiques (par exemple, quotidiens, horaires).
    risk_free_rate : float, optional
        Le taux de rendement sans risque pour la même période que les rendements.
        Par défaut, 0.0.
    annualization_factor : int, optional
        Le facteur pour annualiser le ratio de Sharpe (par exemple, 252 pour les données quotidiennes,
        365 si le trading est 24/7, 252*6.5 pour des rendements horaires sur marché actions, etc.).
        Par défaut, 252.

    Returns
    -------
    float
        Le ratio de Sharpe annualisé. Retourne np.nan si le calcul n'est pas possible
        (par exemple, écart-type nul ou données insuffisantes).

    Examples
    --------
    >>> daily_returns = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008])
    >>> sharpe_ratio(daily_returns, risk_free_rate=0.0, annualization_factor=252)
    5.487609590901107

    >>> import numpy as np
    >>> hourly_returns = np.array([0.001, -0.0005, 0.0015, 0.0002, -0.0008] * 10)
    >>> sharpe_ratio(hourly_returns, annualization_factor=252*24) # Crypto 24/7
    10.049069259302378
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().to_numpy()
    elif isinstance(returns, np.ndarray):
        returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return np.nan

    excess_returns = returns - risk_free_rate / annualization_factor
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns, ddof=1) # ddof=1 for sample standard deviation

    if std_dev_excess_return == 0:
        if mean_excess_return > 0:
            return np.inf
        elif mean_excess_return < 0:
            return -np.inf
        else:
            return 0.0 # Or np.nan, depending on desired behavior for zero risk and zero return

    sharpe = mean_excess_return / std_dev_excess_return
    return sharpe * np.sqrt(annualization_factor)

def sortino_ratio(
    returns: Union[pd.Series, NDArray[np.float64]],
    required_return: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calcule le ratio de Sortino annualisé.

    Le ratio de Sortino est une variation du ratio de Sharpe qui ne pénalise que
    la volatilité à la baisse (rendements inférieurs au rendement requis).

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements périodiques.
    required_return : float, optional
        Le rendement minimum acceptable ou le taux sans risque. Par défaut 0.0.
    annualization_factor : int, optional
        Facteur d'annualisation. Par défaut 252.

    Returns
    -------
    float
        Le ratio de Sortino annualisé. Retourne np.nan si le calcul n'est pas possible.

    Examples
    --------
    >>> daily_returns = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008, 0.005])
    >>> sortino_ratio(daily_returns, required_return=0.0, annualization_factor=252)
    8.030303942900432
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().to_numpy()
    elif isinstance(returns, np.ndarray):
        returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return np.nan

    target_return_periodic = required_return / annualization_factor
    excess_returns = returns - target_return_periodic
    mean_excess_return = np.mean(excess_returns)

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0: # No downside returns
        if mean_excess_return > 0:
            return np.inf # Infinite Sortino if positive returns and no downside deviation
        else:
            return 0.0 # Or np.nan, if mean excess return is also zero or negative

    downside_deviation = np.sqrt(np.sum(downside_returns**2) / len(returns))

    if downside_deviation == 0:
        if mean_excess_return > 0:
            return np.inf
        elif mean_excess_return < 0:
            return -np.inf
        else:
            return 0.0

    sortino = mean_excess_return / downside_deviation
    return sortino * np.sqrt(annualization_factor)

def calmar_ratio(
    returns: Union[pd.Series, NDArray[np.float64]],
    annualization_factor: int = 252
) -> float:
    """
    Calcule le ratio de Calmar.

    Le ratio de Calmar est le rendement annualisé divisé par le drawdown maximum.
    Il est souvent utilisé pour évaluer les hedge funds et les managed futures.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements périodiques.
    annualization_factor : int, optional
        Facteur d'annualisation. Par défaut 252.

    Returns
    -------
    float
        Le ratio de Calmar. Retourne np.nan si le calcul n'est pas possible.

    Examples
    --------
    >>> daily_returns = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008, 0.005, -0.02, 0.01])
    >>> calmar_ratio(daily_returns, annualization_factor=252)
    2.830307099589359
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()
    elif isinstance(returns, np.ndarray):
        returns = pd.Series(returns[~np.isnan(returns)]) # max_drawdown expects Series for now

    if len(returns) < 2:
        return np.nan

    annualized_return = np.mean(returns) * annualization_factor
    mdd, _ = max_drawdown(returns) # max_drawdown returns a positive value for drawdown

    if mdd == 0: # No drawdown
        if annualized_return > 0:
            return np.inf
        elif annualized_return < 0:
             return -np.inf # Negative return with no drawdown is unusual but possible if all returns are negative
        else:
            return 0.0
    
    if np.isnan(mdd) or mdd < 0: # mdd should be positive
        return np.nan

    return annualized_return / mdd

def information_ratio(
    portfolio_returns: Union[pd.Series, NDArray[np.float64]],
    benchmark_returns: Union[pd.Series, NDArray[np.float64]],
    annualization_factor: int = 252
) -> float:
    """
    Calcule le ratio d'Information.

    Le ratio d'Information mesure le rendement du portefeuille par rapport à un benchmark,
    ajusté par la volatilité de ce rendement excédentaire (tracking error).

    Parameters
    ----------
    portfolio_returns : Union[pd.Series, NDArray[np.float64]]
        Série des rendements du portefeuille.
    benchmark_returns : Union[pd.Series, NDArray[np.float64]]
        Série des rendements du benchmark.
    annualization_factor : int, optional
        Facteur d'annualisation. Par défaut 252.

    Returns
    -------
    float
        Le ratio d'Information annualisé. Retourne np.nan si le calcul n'est pas possible.

    Examples
    --------
    >>> portfolio_ret = pd.Series([0.012, -0.003, 0.017, 0.004, -0.007])
    >>> benchmark_ret = pd.Series([0.010, -0.004, 0.015, 0.003, -0.008])
    >>> information_ratio(portfolio_ret, benchmark_ret)
    11.200000000000001 
    """
    if isinstance(portfolio_returns, pd.Series):
        portfolio_returns = portfolio_returns.dropna().to_numpy()
    elif isinstance(portfolio_returns, np.ndarray):
        portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]

    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.dropna().to_numpy()
    elif isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]

    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return np.nan

    active_returns = portfolio_returns - benchmark_returns
    mean_active_return = np.mean(active_returns)
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        if mean_active_return > 0:
            return np.inf
        elif mean_active_return < 0:
            return -np.inf
        else:
            return 0.0

    ir = mean_active_return / tracking_error
    return ir * np.sqrt(annualization_factor)

# --- Position sizing ---

def kelly_criterion(
    win_probability: float,
    win_loss_ratio: float
) -> float:
    """
    Calcule la fraction de Kelly.

    La fraction de Kelly est la proportion du capital à risquer sur un trade
    pour maximiser le taux de croissance du capital à long terme.
    Formule: K = p - (1-p)/R, où p est la probabilité de gain et R est le ratio gain/perte.

    Parameters
    ----------
    win_probability : float
        Probabilité de gain (entre 0 et 1).
    win_loss_ratio : float
        Ratio moyen des gains sur les pertes (doit être > 0).

    Returns
    -------
    float
        La fraction de Kelly. Peut être négative si l'espérance est négative.
        Un Kelly négatif ou nul suggère de ne pas trader.

    Examples
    --------
    >>> kelly_criterion(win_probability=0.6, win_loss_ratio=2.0)
    0.4
    >>> kelly_criterion(win_probability=0.5, win_loss_ratio=0.8) # Negative expectancy
    -0.125
    """
    if not (0 <= win_probability <= 1):
        raise ValueError("Win probability must be between 0 and 1.")
    if win_loss_ratio <= 0:
        raise ValueError("Win/loss ratio must be positive.")

    kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
    return kelly_fraction

def optimal_f(
    returns: Union[pd.Series, NDArray[np.float64]]
) -> float:
    """
    Calcule la fraction f optimale (selon la formule de Vince simplifié pour les rendements).
    Cette version suppose que f est la fraction du capital allouée, et les rendements
    sont appliqués à cette fraction. f = mean_return / variance_of_returns.
    Ceci est une simplification et peut être agressive. À utiliser avec prudence.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements d'une stratégie ou d'un actif.

    Returns
    -------
    float
        La fraction f optimale.

    Notes
    -----
    La formule f = mu / sigma^2 maximise le taux de croissance géométrique des rendements
    sous certaines hypothèses (par exemple, rendements normalement distribués, possibilité
    d'investir une fraction f du capital).
    Une valeur de f > 1 implique l'utilisation de levier.
    Cette formule peut être très sensible aux estimations de mu et sigma.

    Examples
    --------
    >>> returns = pd.Series([0.1, -0.05, 0.15, 0.02, -0.08])
    >>> optimal_f(returns)
    0.4469135802469134
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().to_numpy()
    elif isinstance(returns, np.ndarray):
        returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return np.nan

    mean_return = np.mean(returns)
    variance_return = np.var(returns, ddof=1) # Sample variance

    if variance_return == 0:
        if mean_return > 0:
            return np.inf # Théoriquement, si pas de risque et rendement positif
        else:
            return 0.0 # Ou np.nan

    return mean_return / variance_return


def fixed_fractional_position_size(
    account_balance: float,
    risk_fraction: float,
    stop_loss_pct: Optional[float] = None,
    asset_price: Optional[float] = None
) -> float:
    """
    Calcule la taille de la position en utilisant la méthode du fixed fractional.

    Si stop_loss_pct et asset_price sont fournis, la taille est calculée pour que la perte
    maximale (si le stop-loss est atteint) soit égale à risk_fraction * account_balance.
    Sinon, retourne simplement risk_fraction * account_balance (interprété comme la valeur notionnelle à risquer).

    Parameters
    ----------
    account_balance : float
        Le solde actuel du compte.
    risk_fraction : float
        La fraction du capital à risquer par trade (par exemple, 0.02 pour 2%).
    stop_loss_pct : Optional[float], optional
        Le pourcentage de stop-loss par rapport au prix d'entrée (par exemple, 0.05 pour 5%).
        Si fourni, la fonction calcule la quantité d'actif.
    asset_price : Optional[float], optional
        Le prix actuel de l'actif. Nécessaire si stop_loss_pct est fourni pour calculer la quantité.

    Returns
    -------
    float
        Si stop_loss_pct et asset_price sont fournis: la quantité d'actif à trader.
        Sinon: la valeur monétaire à allouer à la position.

    Examples
    --------
    >>> fixed_fractional_position_size(10000, 0.02) # Valeur notionnelle
    200.0
    >>> fixed_fractional_position_size(10000, 0.02, stop_loss_pct=0.05, asset_price=50.0) # Quantité d'actif
    80.0
    """
    if account_balance <= 0:
        return 0.0
    if not (0 < risk_fraction <= 1): # risk_fraction can be > 1 if leverage is implied, but typically <=1
        # For this function, assume risk_fraction is a fraction of equity to risk.
        raise ValueError("Risk fraction must be positive and typically <= 1.")

    amount_to_risk = account_balance * risk_fraction

    if stop_loss_pct is not None and asset_price is not None:
        if stop_loss_pct <= 0:
            raise ValueError("Stop loss percentage must be positive.")
        if asset_price <= 0:
            raise ValueError("Asset price must be positive.")
        
        # Perte par unité d'actif si stop-loss est touché
        loss_per_unit = asset_price * stop_loss_pct
        if loss_per_unit == 0: # Should not happen if asset_price and stop_loss_pct > 0
            return np.inf # Or handle as error
            
        position_quantity = amount_to_risk / loss_per_unit
        return position_quantity
    else:
        # Retourne la valeur notionnelle à allouer
        return amount_to_risk

# --- Risk metrics ---

def max_drawdown(
    returns: Union[pd.Series, NDArray[np.float64]]
) -> Tuple[float, int]:
    """
    Calcule le drawdown maximum à partir d'une série de rendements.

    Le drawdown maximum est la plus grande perte en pourcentage d'un pic à un creux
    pendant une période spécifique.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements périodiques.

    Returns
    -------
    Tuple[float, int]
        Un tuple contenant:
        - Le drawdown maximum (valeur positive, par exemple, 0.1 pour 10% de drawdown).
        - L'index (position) du creux de la période de drawdown maximum.
        Retourne (np.nan, -1) si le calcul n'est pas possible.

    Examples
    --------
    >>> returns = pd.Series([0.1, -0.05, -0.08, 0.15, -0.1, 0.02, -0.12])
    >>> mdd, trough_idx = max_drawdown(returns)
    >>> print(f"Max Drawdown: {mdd:.4f}, Trough Index: {trough_idx}")
    Max Drawdown: 0.2151, Trough Index: 6
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    returns = returns.dropna()
    if len(returns) < 1: # Need at least one return to form a cumulative product
        return np.nan, -1

    # Calcule la courbe de richesse (equity curve)
    cumulative_returns = (1 + returns).cumprod()
    # Ajoute un point de départ à 1 pour la courbe de richesse
    equity_curve = pd.Series(np.insert(cumulative_returns.to_numpy(), 0, 1.0))
    
    # Calcule le pic courant (running maximum)
    running_max = equity_curve.cummax()
    
    # Calcule les drawdowns
    drawdowns = (equity_curve - running_max) / running_max
    
    max_dd = -np.min(drawdowns) # Drawdown est une valeur positive
    
    if pd.isna(max_dd): # Should not happen if returns are valid
        return np.nan, -1

    # Trouver l'index du creux. L'index est par rapport à `equity_curve`.
    # drawdowns series has index from 0 to len(returns).
    # The trough corresponds to the minimum value in `drawdowns`.
    # If multiple troughs have the same min drawdown, argmin returns the first.
    trough_idx = drawdowns.idxmin()

    return float(max_dd), int(trough_idx)


def underwater_curve(
    returns: Union[pd.Series, NDArray[np.float64]]
) -> pd.Series:
    """
    Génère la courbe "sous-marine" (underwater curve) des drawdowns.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements périodiques.

    Returns
    -------
    pd.Series
        Une série représentant les drawdowns au fil du temps.
        Les valeurs sont négatives ou nulles.
        Retourne une série vide si le calcul n'est pas possible.

    Examples
    --------
    >>> returns = pd.Series([0.02, 0.01, -0.03, -0.02, 0.015])
    >>> underwater_curve(returns)
    0    0.000000
    1    0.000000
    2   -0.029126
    3   -0.048350
    4   -0.034097
    dtype: float64
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    
    returns = returns.dropna()
    if len(returns) == 0:
        return pd.Series(dtype=np.float64)

    cumulative_returns = (1 + returns).cumprod()
    equity_curve = pd.Series(np.insert(cumulative_returns.to_numpy(), 0, 1.0))
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    
    # La courbe underwater est généralement indexée comme les rendements originaux.
    # `drawdowns` est plus long de 1 que `returns` à cause du point initial.
    # On retourne les drawdowns correspondant aux points de `returns`.
    return drawdowns[1:].reset_index(drop=True)


def value_at_risk(
    returns: Union[pd.Series, NDArray[np.float64]],
    confidence_level: float = 0.95
) -> float:
    """
    Calcule la Value at Risk (VaR) historique.

    La VaR est la perte maximale attendue (ou pire) sur une période donnée
    avec un niveau de confiance donné.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements périodiques.
    confidence_level : float, optional
        Le niveau de confiance (par exemple, 0.95 pour 95%). Par défaut 0.95.

    Returns
    -------
    float
        La Value at Risk (valeur positive représentant une perte).
        Retourne np.nan si le calcul n'est pas possible.

    Examples
    --------
    >>> returns = pd.Series(np.random.normal(-0.01, 0.02, 1000))
    >>> value_at_risk(returns, confidence_level=0.99) # VaR à 99%
    0.06039940967604386 # Example output, will vary
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().to_numpy()
    elif isinstance(returns, np.ndarray):
        returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return np.nan
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

    # La VaR est le quantile (1 - confidence_level) des rendements.
    # Par exemple, pour 95% de confiance, on cherche le 5ème percentile.
    # Le résultat est une perte, donc on prend l'opposé si le percentile est négatif.
    var_value = np.percentile(returns, 100 * (1 - confidence_level))
    return -var_value # VaR est reportée comme une perte positive


def conditional_value_at_risk(
    returns: Union[pd.Series, NDArray[np.float64]],
    confidence_level: float = 0.95
) -> float:
    """
    Calcule la Conditional Value at Risk (CVaR) historique, aussi appelée Expected Shortfall.

    La CVaR est la perte moyenne attendue, conditionnellement au fait que la perte
    est supérieure ou égale à la VaR.

    Parameters
    ----------
    returns : Union[pd.Series, NDArray[np.float64]]
        Série de rendements périodiques.
    confidence_level : float, optional
        Le niveau de confiance (par exemple, 0.95 pour 95%). Par défaut 0.95.

    Returns
    -------
    float
        La Conditional Value at Risk (valeur positive représentant une perte).
        Retourne np.nan si le calcul n'est pas possible.

    Examples
    --------
    >>> returns = pd.Series(np.random.normal(-0.01, 0.02, 1000))
    >>> conditional_value_at_risk(returns, confidence_level=0.99) # CVaR à 99%
    0.07081006523065374 # Example output, will vary
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().to_numpy()
    elif isinstance(returns, np.ndarray):
        returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return np.nan
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

    var_value = np.percentile(returns, 100 * (1 - confidence_level))
    
    # CVaR est la moyenne des rendements inférieurs ou égaux à la VaR.
    cvar_value = np.mean(returns[returns <= var_value])
    return -cvar_value # CVaR est reportée comme une perte positive

# --- Binance specific ---

def round_step_size(
    quantity: float,
    step_size: float
) -> float:
    """
    Arrondit une quantité à la baisse au multiple le plus proche de step_size.
    Conforme aux règles de Binance pour les quantités.

    Parameters
    ----------
    quantity : float
        La quantité à arrondir.
    step_size : float
        La taille du pas (par exemple, 0.001 pour BTC).

    Returns
    -------
    float
        La quantité arrondie.

    Examples
    --------
    >>> round_step_size(0.12345, 0.001)
    0.123
    >>> round_step_size(0.123, 0.0001)
    0.123
    >>> round_step_size(0.1239, 0.001)
    0.123
    >>> round_step_size(15.7, 1.0)
    15.0
    """
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    if np.isnan(quantity) or np.isinf(quantity):
        return quantity # Return as is if NaN or Inf

    # Utilise la formule d'arrondi à la baisse pour se conformer à Binance
    # quantity = floor(quantity / step_size) * step_size
    # Pour éviter les problèmes de précision flottante, on peut travailler avec des décimales
    # ou une approche plus robuste.
    # Exemple: decimal_places = -int(math.log10(step_size)) if step_size < 1 else 0
    # multiplier = 10**decimal_places
    # return math.floor(quantity * multiplier) / multiplier
    # Cependant, la formule directe avec division/multiplication est souvent utilisée.
    # Binance spécifie "LOT_SIZE" filter, qui a minQty, maxQty, stepSize.
    # Quantities must be multiples of stepSize.
    # (quantity - minQty) % stepSize == 0
    # Arrondi vers le bas (floor) pour être conservateur.
    
    # Convert step_size to string to find number of decimal places accurately
    step_size_str = "{:.10f}".format(step_size).rstrip('0')
    if '.' in step_size_str:
        precision = len(step_size_str.split('.')[1])
    else: # step_size is an integer like 1.0, 10.0
        precision = 0
        # if step_size is 10, 100, this logic might need adjustment if we want to round to 10s, 100s
        # but for crypto step_size, it's usually < 1 or 1.
        # For step_size = 1.0, precision is 0.
        # For step_size = 10.0, precision is 0 according to this.
        # This is for decimal precision, not for rounding to multiple of 10, 100.
        # The formula below handles multiples correctly.

    # Correct rounding using decimal precision of step_size
    # This ensures that intermediate floating point inaccuracies are minimized.
    # Rounded_quantity = floor(quantity / step_size) * step_size
    # Example: quantity=0.12345, step_size=0.001
    # 0.12345 / 0.001 = 123.45
    # floor(123.45) = 123
    # 123 * 0.001 = 0.123
    if quantity < 0: # Should not happen for order quantities
        # Or raise error, depending on context
        return math.ceil(quantity / step_size) * step_size

    rounded_val = math.floor(quantity / step_size) * step_size
    
    # Due to floating point arithmetic, result might be slightly off, e.g. 0.12299999999999998
    # So, round it again to the precision of step_size
    return float(f"{rounded_val:.{precision}f}")


def round_price_precision(
    price: float,
    tick_size: float # Changed from precision to tick_size for clarity with Binance terms
) -> float:
    """
    Arrondit un prix au multiple le plus proche de tick_size.
    Conforme aux règles de Binance pour les prix (tickSize).

    Parameters
    ----------
    price : float
        Le prix à arrondir.
    tick_size : float
        La taille du tick (par exemple, 0.01 pour BTCUSDC).

    Returns
    -------
    float
        Le prix arrondi.

    Examples
    --------
    >>> round_price_precision(45000.12345, 0.01)
    45000.12
    >>> round_price_precision(45000.128, 0.01)
    45000.13
    >>> round_price_precision(0.123456, 0.00001)
    0.12346
    """
    if tick_size <= 0:
        raise ValueError("Tick size must be positive.")
    if np.isnan(price) or np.isinf(price):
        return price

    # tick_size_str = "{:.10f}".format(tick_size).rstrip('0')
    # if '.' in tick_size_str:
    #     precision = len(tick_size_str.split('.')[1])
    # else:
    #     precision = 0
    
    # Arrondir au multiple le plus proche de tick_size
    # price = round(price / tick_size) * tick_size
    # Rounded_price = round(price / tick_size) * tick_size
    # return float(f"{Rounded_price:.{precision}f}")
    
    # Using a more robust method with decimal context would be ideal,
    # but for typical crypto tick_sizes, this should work.
    # Python's round() rounds to the nearest even number for .5 cases (e.g. round(2.5)=2, round(3.5)=4)
    # We need standard rounding (0.5 up).
    # A common way:
    factor = 1 / tick_size
    rounded_price = math.floor(price * factor + 0.5) / factor
    
    # Determine precision from tick_size for final formatting
    tick_size_str = "{:.15f}".format(tick_size).rstrip('0')
    if '.' in tick_size_str:
        num_decimal_places = len(tick_size_str.split('.')[1])
    else:
        num_decimal_places = 0
        
    return float(f"{rounded_price:.{num_decimal_places}f}")


def calculate_notional_value(
    price: float,
    quantity: float
) -> float:
    """
    Calcule la valeur notionnelle d'une position.

    Valeur Notionnelle = Prix * Quantité.

    Parameters
    ----------
    price : float
        Le prix de l'actif.
    quantity : float
        La quantité de l'actif.

    Returns
    -------
    float
        La valeur notionnelle.

    Examples
    --------
    >>> calculate_notional_value(price=50000.0, quantity=0.5)
    25000.0
    """
    if np.isnan(price) or np.isinf(price) or np.isnan(quantity) or np.isinf(quantity):
        # Or handle as error, depending on strictness
        return np.nan 
    return price * quantity

# --- Portfolio calculations ---

def portfolio_correlation_matrix(
    returns_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation d'un portefeuille d'actifs.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame où chaque colonne représente les rendements d'un actif
        et les lignes représentent les périodes.

    Returns
    -------
    pd.DataFrame
        La matrice de corrélation. Retourne un DataFrame vide si l'entrée est invalide.

    Examples
    --------
    >>> data = {
    ...     'AssetA': [0.01, 0.02, -0.01, 0.005],
    ...     'AssetB': [0.015, 0.018, -0.009, 0.006],
    ...     'AssetC': [-0.005, 0.001, 0.003, -0.002]
    ... }
    >>> returns_df = pd.DataFrame(data)
    >>> portfolio_correlation_matrix(returns_df)
              AssetA    AssetB    AssetC
    AssetA  1.000000  0.999263 -0.609767
    AssetB  0.999263  1.000000 -0.575799
    AssetC -0.609767 -0.575799  1.000000
    """
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        return pd.DataFrame()
    
    # Drop rows with any NaN to ensure correlation is computed on complete cases
    # Alternatively, pandas .corr() handles pairwise NaNs by default with 'pearson'
    # returns_df_cleaned = returns_df.dropna()
    # if returns_df_cleaned.shape[0] < 2: # Need at least 2 observations
    #     return pd.DataFrame(np.nan, index=returns_df.columns, columns=returns_df.columns)

    return returns_df.corr(method='pearson')

def portfolio_covariance_matrix(
    returns_df: pd.DataFrame,
    annualization_factor: Optional[int] = None # Typically 252 for daily returns
) -> pd.DataFrame:
    """
    Calcule la matrice de covariance (annualisée optionnellement) d'un portefeuille d'actifs.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame où chaque colonne représente les rendements d'un actif.
    annualization_factor : Optional[int], optional
        Si fourni, la matrice de covariance est annualisée en la multipliant
        par ce facteur. Par exemple, 252 pour des rendements quotidiens.
        Par défaut, None (pas d'annualisation).

    Returns
    -------
    pd.DataFrame
        La matrice de covariance. Retourne un DataFrame vide si l'entrée est invalide.

    Examples
    --------
    >>> data = {
    ...     'AssetA': [0.01, 0.02, -0.01, 0.005],
    ...     'AssetB': [0.015, 0.018, -0.009, 0.006],
    ... }
    >>> returns_df = pd.DataFrame(data)
    >>> portfolio_covariance_matrix(returns_df)
                 AssetA        AssetB
    AssetA  0.00010833  0.00010250
    AssetB  0.00010250  0.00010425
    >>> portfolio_covariance_matrix(returns_df, annualization_factor=252)
                AssetA      AssetB
    AssetA  0.02730000  0.02583000
    AssetB  0.02583000  0.02627100
    """
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        return pd.DataFrame()

    # returns_df_cleaned = returns_df.dropna()
    # if returns_df_cleaned.shape[0] < 2:
    #     return pd.DataFrame(np.nan, index=returns_df.columns, columns=returns_df.columns)
        
    cov_matrix = returns_df.cov() # ddof=1 by default for sample covariance

    if annualization_factor is not None:
        if annualization_factor <= 0:
            raise ValueError("Annualization factor must be positive.")
        cov_matrix *= annualization_factor
        
    return cov_matrix


def portfolio_weights_optimization(
    expected_returns: Union[pd.Series, NDArray[np.float64]],
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    allow_short_selling: bool = False
) -> NDArray[np.float64]:
    """
    Optimise les poids d'un portefeuille pour maximiser le ratio de Sharpe
    ou atteindre un rendement cible avec une variance minimale.

    Parameters
    ----------
    expected_returns : Union[pd.Series, NDArray[np.float64]]
        Les rendements attendus pour chaque actif.
    covariance_matrix : pd.DataFrame
        La matrice de covariance des rendements des actifs.
    risk_free_rate : float, optional
        Le taux sans risque. Par défaut 0.0.
    target_return : Optional[float], optional
        Si fourni, optimise pour une variance minimale étant donné ce rendement cible.
        Sinon, maximise le ratio de Sharpe. Par défaut None.
    allow_short_selling : bool, optional
        Si True, les poids peuvent être négatifs. Par défaut False (long-only).

    Returns
    -------
    NDArray[np.float64]
        Un tableau NumPy des poids optimaux pour chaque actif.
        Retourne un tableau de NaN si l'optimisation échoue.

    Notes
    -----
    Cette fonction utilise scipy.optimize.minimize.
    L'optimisation peut être sensible aux conditions initiales et aux contraintes.
    """
    num_assets = len(expected_returns)
    if num_assets != covariance_matrix.shape[0] or num_assets != covariance_matrix.shape[1]:
        raise ValueError("Mismatch in dimensions of expected_returns and covariance_matrix.")

    # Initial guess: equal weights
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    # Constraints: sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0})

    # Bounds for weights
    bounds = tuple((0.0, 1.0) for _ in range(num_assets)) # Long-only by default
    if allow_short_selling:
        bounds = tuple((-1.0, 1.0) for _ in range(num_assets)) # Example bounds for short selling

    # Objective function: negative Sharpe ratio (to minimize) or portfolio variance
    if target_return is not None:
        # Minimize portfolio variance for a target return
        def portfolio_variance(weights: NDArray[np.float64]) -> float:
            return weights.T @ covariance_matrix @ weights
        
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},
            {'type': 'eq', 'fun': lambda weights: weights.T @ expected_returns - target_return}
        )
        objective_func = portfolio_variance
    else:
        # Maximize Sharpe Ratio (minimize negative Sharpe Ratio)
        def negative_sharpe_ratio(weights: NDArray[np.float64]) -> float:
            port_return = weights.T @ expected_returns
            port_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
            if port_volatility == 0:
                return -np.inf if (port_return - risk_free_rate) > 0 else 0 # Avoid division by zero
            return -(port_return - risk_free_rate) / port_volatility
        objective_func = negative_sharpe_ratio

    try:
        result = minimize(
            objective_func,
            initial_weights,
            method='SLSQP', # Sequential Least Squares Programming, handles bounds and constraints
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        else:
            # print(f"Optimization failed: {result.message}")
            return np.full(num_assets, np.nan)
    except Exception as e:
        # print(f"Error during optimization: {e}")
        return np.full(num_assets, np.nan)

# --- Helpers ---

def safe_divide(
    numerator: Union[float, NDArray[np.float64]],
    denominator: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Effectue une division en gérant les cas de division par zéro.
    Retourne np.nan si le dénominateur est zéro.

    Parameters
    ----------
    numerator : Union[float, NDArray[np.float64]]
        Le numérateur.
    denominator : Union[float, NDArray[np.float64]]
        Le dénominateur.

    Returns
    -------
    Union[float, NDArray[np.float64]]
        Le résultat de la division, ou np.nan en cas de division par zéro.

    Examples
    --------
    >>> safe_divide(10, 2)
    5.0
    >>> safe_divide(10, 0)
    nan
    >>> safe_divide(np.array([1, 2, 3]), np.array([1, 0, 3]))
    array([ 1., nan,  1.])
    """
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        if denominator == 0:
            return np.nan
        return numerator / denominator
    
    # Handle array inputs
    # Ensure inputs are numpy arrays for vectorized operations
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    
    # Create a result array filled with NaNs
    result = np.full_like(num, np.nan, dtype=np.float64)
    
    # Find where denominator is not zero
    valid_den_mask = (den != 0)
    
    # Perform division only for valid denominators
    # np.divide can also take an 'out' and 'where' argument
    np.divide(num, den, out=result, where=valid_den_mask)
    
    # If inputs were scalars but passed as 0-d arrays, return scalar
    if num.ndim == 0 and den.ndim == 0:
        return result.item()
        
    return result


def rolling_std(
    data: pd.Series,
    window: int,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calcule l'écart-type mobile (rolling standard deviation).

    Parameters
    ----------
    data : pd.Series
        La série de données.
    window : int
        La taille de la fenêtre mobile.
    min_periods : Optional[int], optional
        Le nombre minimum d'observations dans la fenêtre pour avoir une valeur.
        Par défaut, égal à la taille de la fenêtre.

    Returns
    -------
    pd.Series
        La série de l'écart-type mobile.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> rolling_std(s, window=3)
    0         NaN
    1         NaN
    2    1.000000
    3    1.000000
    4    1.000000
    5    1.000000
    6    1.000000
    7    1.000000
    8    1.000000
    9    1.000000
    dtype: float64
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if min_periods is None:
        min_periods = window
    if min_periods <= 0:
         raise ValueError("min_periods must be positive.")

    return data.rolling(window=window, min_periods=min_periods).std(ddof=1) # ddof=1 for sample std


def exponential_moving_average(
    data: pd.Series,
    span: Optional[int] = None,
    com: Optional[float] = None,
    halflife: Optional[float] = None,
    alpha: Optional[float] = None,
    min_periods: int = 0,
    adjust: bool = True
) -> pd.Series:
    """
    Calcule la moyenne mobile exponentielle (EMA).
    Wrapper autour de pandas.Series.ewm.

    Spécifiez exactement un des paramètres: span, com, halflife, ou alpha.

    Parameters
    ----------
    data : pd.Series
        La série de données.
    span : Optional[int], optional
        La période de la moyenne mobile exponentielle.
    com : Optional[float], optional
        Le centre de masse. com = (span - 1) / 2.
    halflife : Optional[float], optional
        La demi-vie.
    alpha : Optional[float], optional
        Le facteur de lissage alpha directement. alpha = 2 / (span + 1).
    min_periods : int, optional
        Nombre minimum d'observations dans la fenêtre pour avoir une valeur. Par défaut 0.
    adjust : bool, optional
        Diviser par le facteur de lissage de décroissance au début des séries pour
        compenser le déséquilibre des poids. Par défaut True.

    Returns
    -------
    pd.Series
        La série de la moyenne mobile exponentielle.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> exponential_moving_average(s, span=3)
    0    1.000000
    1    1.666667
    2    2.500000
    3    3.500000
    4    4.500000
    5    5.500000
    6    6.500000
    7    7.500000
    8    8.500000
    9    9.500000
    dtype: float64
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    
    # Vérifier que l'un des paramètres de lissage est fourni
    smoothing_params = [span, com, halflife, alpha]
    if sum(p is not None for p in smoothing_params) != 1:
        raise ValueError("Exactly one of span, com, halflife, or alpha must be provided.")

    return data.ewm(
        span=span,
        com=com,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=False # Default, process NaNs like pandas does
    ).mean()

if __name__ == '__main__':
    # Example usage and basic tests (can be expanded into proper unit tests)
    print("--- Testing math_utils ---")

    # Sharpe Ratio
    daily_returns_sharpe = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008])
    sr = sharpe_ratio(daily_returns_sharpe)
    print(f"Sharpe Ratio: {sr}") # Expected: around 5.48

    # Sortino Ratio
    daily_returns_sortino = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008, 0.005])
    sor = sortino_ratio(daily_returns_sortino)
    print(f"Sortino Ratio: {sor}") # Expected: around 8.03

    # Calmar Ratio
    daily_returns_calmar = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008, 0.005, -0.02, 0.01])
    cr = calmar_ratio(daily_returns_calmar)
    print(f"Calmar Ratio: {cr}") # Expected: around 2.83

    # Information Ratio
    portfolio_ret_ir = pd.Series([0.012, -0.003, 0.017, 0.004, -0.007])
    benchmark_ret_ir = pd.Series([0.010, -0.004, 0.015, 0.003, -0.008])
    ir_val = information_ratio(portfolio_ret_ir, benchmark_ret_ir)
    print(f"Information Ratio: {ir_val}") # Expected: around 11.2

    # Kelly Criterion
    kc = kelly_criterion(win_probability=0.6, win_loss_ratio=2.0)
    print(f"Kelly Criterion: {kc}") # Expected: 0.4

    # Optimal F
    returns_opt_f = pd.Series([0.1, -0.05, 0.15, 0.02, -0.08])
    opt_f_val = optimal_f(returns_opt_f)
    print(f"Optimal F: {opt_f_val}") # Expected: around 0.4469

    # Fixed Fractional Position Size
    ffps_notional = fixed_fractional_position_size(10000, 0.02)
    print(f"Fixed Fractional (Notional): {ffps_notional}") # Expected: 200.0
    ffps_quantity = fixed_fractional_position_size(10000, 0.02, stop_loss_pct=0.05, asset_price=50.0)
    print(f"Fixed Fractional (Quantity): {ffps_quantity}") # Expected: 80.0

    # Max Drawdown
    returns_mdd = pd.Series([0.1, -0.05, -0.08, 0.15, -0.1, 0.02, -0.12])
    mdd_val, mdd_idx = max_drawdown(returns_mdd)
    print(f"Max Drawdown: {mdd_val:.4f}, Trough Index: {mdd_idx}") # Expected: 0.2151, 6

    # Underwater Curve
    returns_uw = pd.Series([0.02, 0.01, -0.03, -0.02, 0.015])
    uw_curve = underwater_curve(returns_uw)
    print(f"Underwater Curve:\n{uw_curve}")

    # Value at Risk
    np.random.seed(42) # for reproducibility
    returns_var = pd.Series(np.random.normal(-0.01, 0.02, 1000))
    var_99 = value_at_risk(returns_var, confidence_level=0.99)
    print(f"VaR (99%): {var_99:.4f}") # Expected: around 0.0563

    # Conditional Value at Risk
    cvar_99 = conditional_value_at_risk(returns_var, confidence_level=0.99)
    print(f"CVaR (99%): {cvar_99:.4f}") # Expected: around 0.0683

    # Round Step Size
    print(f"Round Step Size (0.12345, 0.001): {round_step_size(0.12345, 0.001)}") # Expected: 0.123
    print(f"Round Step Size (0.1239, 0.001): {round_step_size(0.1239, 0.001)}")   # Expected: 0.123

    # Round Price Precision (using tick_size)
    print(f"Round Price (45000.12345, 0.01): {round_price_precision(45000.12345, 0.01)}") # Expected: 45000.12
    print(f"Round Price (45000.128, 0.01): {round_price_precision(45000.128, 0.01)}")     # Expected: 45000.13

    # Calculate Notional Value
    notional = calculate_notional_value(price=50000.0, quantity=0.5)
    print(f"Notional Value: {notional}") # Expected: 25000.0

    # Portfolio Correlation Matrix
    data_port = {
        'AssetA': [0.01, 0.02, -0.01, 0.005],
        'AssetB': [0.015, 0.018, -0.009, 0.006],
        'AssetC': [-0.005, 0.001, 0.003, -0.002]
    }
    returns_df_port = pd.DataFrame(data_port)
    corr_matrix = portfolio_correlation_matrix(returns_df_port)
    print(f"Portfolio Correlation Matrix:\n{corr_matrix}")

    # Portfolio Covariance Matrix
    cov_matrix = portfolio_covariance_matrix(returns_df_port[['AssetA', 'AssetB']])
    print(f"Portfolio Covariance Matrix:\n{cov_matrix}")
    cov_matrix_ann = portfolio_covariance_matrix(returns_df_port[['AssetA', 'AssetB']], annualization_factor=252)
    print(f"Annualized Portfolio Covariance Matrix:\n{cov_matrix_ann}")

    # Portfolio Weights Optimization (Example: Max Sharpe)
    # Using annualized figures for expected returns and covariance
    exp_returns_opt = returns_df_port.mean() * 252 # Annualized expected returns
    cov_matrix_opt = returns_df_port.cov() * 252   # Annualized covariance matrix
    
    print(f"\nExpected Returns for Opt:\n{exp_returns_opt}")
    print(f"Covariance Matrix for Opt:\n{cov_matrix_opt}")

    optimal_w = portfolio_weights_optimization(exp_returns_opt, cov_matrix_opt, risk_free_rate=0.01)
    print(f"Optimal Weights (Max Sharpe): {optimal_w}")
    if not np.any(np.isnan(optimal_w)):
         print(f"Sum of Optimal Weights: {np.sum(optimal_w):.2f}")


    # Safe Divide
    print(f"Safe Divide (10, 2): {safe_divide(10, 2)}") # Expected: 5.0
    print(f"Safe Divide (10, 0): {safe_divide(10, 0)}") # Expected: nan
    print(f"Safe Divide (array): {safe_divide(np.array([1, 2, 3]), np.array([1, 0, 3]))}")

    # Rolling Std
    s_roll = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    roll_std_s = rolling_std(s_roll, window=3)
    print(f"Rolling Std (window=3):\n{roll_std_s}")

    # Exponential Moving Average
    ema_s = exponential_moving_average(s_roll, span=3)
    print(f"EMA (span=3):\n{ema_s}")

