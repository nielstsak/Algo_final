# src/backtesting/slippage_model.py
import pandas as pd
import numpy as np
from typing import Union, Optional
from loguru import logger

class SlippageModel:
    """
    Classe de base pour modéliser le slippage.
    Le slippage est la différence entre le prix attendu d'un trade et le prix auquel le trade est effectivement exécuté.
    """
    def calculate_slippage(
        self,
        data: pd.DataFrame, # Données OHLCV ou de prix
        symbol: str,
        trade_on_close: bool = False, # Si True, le trade est supposé être au prix de clôture (moins de slippage)
                                     # Si False, le trade est au prix d'ouverture suivant (plus de slippage potentiel)
        order_size: Optional[Union[float, pd.Series]] = None, # Taille de l'ordre (en actif de base)
        is_market_order: Union[bool, pd.Series] = True # Si True, c'est un ordre au marché (plus de slippage)
    ) -> Union[float, pd.Series]:
        """
        Calcule le taux de slippage.
        Doit être surchargée par les classes enfants pour des modèles plus complexes.

        Args:
            data: DataFrame contenant les données de prix ou OHLCV.
                  Peut être utilisé pour des modèles basés sur la volatilité ou le volume.
            symbol: Symbole de la paire de trading.
            trade_on_close: Indique si le trade est exécuté à la clôture ou à l'ouverture suivante.
            order_size: Taille de l'ordre. Peut influencer le slippage.
            is_market_order: Si True, l'ordre est au marché. Les ordres limites subissent moins de slippage.

        Returns:
            Taux de slippage (ex: 0.0005 pour 0.05%).
            Peut être un float (slippage fixe) ou une pd.Series (slippage variable).
        """
        raise NotImplementedError("La méthode calculate_slippage doit être implémentée par la sous-classe.")

class FixedSlippageModel(SlippageModel):
    """
    Modèle de slippage simple avec un taux fixe.
    """
    def __init__(self, fixed_rate: float = 0.0005): # 0.05% par défaut
        if not (0 <= fixed_rate < 1):
            raise ValueError("Le taux de slippage fixe doit être compris entre 0 et 1 (exclusif de 1).")
        self.fixed_rate = fixed_rate
        logger.info(f"FixedSlippageModel initialized with rate: {self.fixed_rate*100:.4f}%")

    def calculate_slippage(
        self,
        data: pd.DataFrame,
        symbol: str,
        trade_on_close: bool = False,
        order_size: Optional[Union[float, pd.Series]] = None,
        is_market_order: Union[bool, pd.Series] = True
    ) -> Union[float, pd.Series]:
        """
        Retourne le taux de slippage fixe.

        Args:
            data, symbol, trade_on_close, order_size, is_market_order: Non utilisés dans ce modèle simple.

        Returns:
            Le taux de slippage fixe.
        """
        return self.fixed_rate

class VolumeBasedSlippageModel(SlippageModel):
    """
    Modèle de slippage qui dépend du volume de trading et de la taille de l'ordre.
    Ceci est une implémentation très simplifiée.
    """
    def __init__(
        self,
        base_slippage_rate: float = 0.0001, # 0.01%
        volume_sensitivity_factor: float = 0.1, # Comment le slippage augmente avec la taille de l'ordre / volume du marché
        avg_daily_volume_period: int = 20 # Période pour calculer le volume quotidien moyen
    ):
        self.base_slippage_rate = base_slippage_rate
        self.volume_sensitivity_factor = volume_sensitivity_factor
        self.avg_daily_volume_period = avg_daily_volume_period
        logger.info(f"VolumeBasedSlippageModel initialized: base_rate={base_slippage_rate*100:.4f}%, sensitivity={volume_sensitivity_factor}")

    def calculate_slippage(
        self,
        data: pd.DataFrame, # Doit contenir 'volume' et 'close'
        symbol: str,
        trade_on_close: bool = False,
        order_size: Optional[Union[float, pd.Series]] = None, # Taille de l'ordre en actif de base
        is_market_order: Union[bool, pd.Series] = True
    ) -> Union[float, pd.Series]:
        """
        Calcule le slippage basé sur le volume.

        Args:
            data: DataFrame avec au moins les colonnes 'volume' et 'close'.
            symbol: Symbole de la paire.
            trade_on_close: Non utilisé directement ici, mais pourrait l'être.
            order_size: Taille de l'ordre en actif de base. Si None, utilise un slippage de base.
            is_market_order: Si False (ordre limite), le slippage peut être réduit.

        Returns:
            Taux de slippage (float ou pd.Series).
        """
        if 'volume' not in data.columns or 'close' not in data.columns:
            logger.warning("Colonnes 'volume' ou 'close' manquantes pour VolumeBasedSlippageModel. Retour au slippage de base.")
            return self.base_slippage_rate

        # Calculer le volume quotidien moyen (approximatif si les données ne sont pas journalières)
        # Pour simplifier, on utilise le volume de la barre actuelle comme proxy de liquidité instantanée
        # Un meilleur modèle utiliserait le volume moyen sur une période.
        # Ici, data['volume'] est le volume de la barre (ex: 1m, 1h).
        # On a besoin du volume de l'actif de base.
        market_liquidity_proxy = data['volume'] # Volume de la barre actuelle

        # Si order_size n'est pas fourni ou si la liquidité est nulle/NaN, retourner le slippage de base
        if order_size is None or (isinstance(market_liquidity_proxy, pd.Series) and (market_liquidity_proxy.isnull().all() or (market_liquidity_proxy == 0).all())):
            return self.base_slippage_rate
        
        # S'assurer que order_size est une série si market_liquidity_proxy l'est
        if isinstance(market_liquidity_proxy, pd.Series) and not isinstance(order_size, pd.Series):
            order_size_series = pd.Series(order_size, index=market_liquidity_proxy.index)
        elif isinstance(order_size, pd.Series):
            order_size_series = order_size
        else: # Les deux sont des scalaires
            order_size_series = pd.Series(order_size) # Pour uniformiser
            market_liquidity_proxy = pd.Series(market_liquidity_proxy, index=order_size_series.index)


        # Ratio de la taille de l'ordre par rapport à la liquidité du marché
        # Remplacer les liquidités nulles ou NaN pour éviter la division par zéro
        market_liquidity_proxy_safe = market_liquidity_proxy.replace(0, np.nan).fillna(method='ffill').fillna(1e-9) # Remplacer 0 par une petite valeur
        
        order_to_liquidity_ratio = (order_size_series / market_liquidity_proxy_safe).fillna(0)

        # Calcul du slippage
        # Formule exemple: slippage = base_rate + sensitivity * (order_size / market_volume_proxy)
        # Le slippage augmente si la taille de l'ordre est grande par rapport au volume disponible.
        slippage = self.base_slippage_rate + self.volume_sensitivity_factor * order_to_liquidity_ratio
        
        # Ajuster pour les ordres limites (moins de slippage)
        if isinstance(is_market_order, pd.Series):
            slippage = np.where(is_market_order, slippage, slippage * 0.5) # Ex: 50% de réduction pour ordres limites
        elif not is_market_order:
            slippage *= 0.5

        # S'assurer que le slippage n'est pas négatif et plafonner si nécessaire
        slippage_final = np.clip(slippage, 0, 0.1) # Plafonner à 10% max pour éviter des valeurs extrêmes

        if isinstance(slippage_final, pd.Series):
            return slippage_final.fillna(self.base_slippage_rate) # Remplir les NaN restants
        return slippage_final if pd.notna(slippage_final) else self.base_slippage_rate


# Exemple d'utilisation:
if __name__ == "__main__":
    fixed_model = FixedSlippageModel(fixed_rate=0.001)
    print(f"Fixed Slippage: {fixed_model.calculate_slippage(None, 'BTCUSDT')*100:.3f}%") # type: ignore

    # Pour VolumeBasedSlippageModel
    idx = pd.date_range('2023-01-01', periods=5, freq='h')
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1200, 800, 1500, 900] # Volume de l'actif de base pour la barre
    }, index=idx)

    volume_model = VolumeBasedSlippageModel(base_slippage_rate=0.0002, volume_sensitivity_factor=0.05)
    
    # Slippage avec taille d'ordre scalaire
    slippage1 = volume_model.calculate_slippage(test_data, 'BTCUSDT', order_size=50) # Ordre de 50 BTC
    print(f"\nVolume Based Slippage (order_size=50):\n{slippage1*100}")

    # Slippage avec taille d'ordre en série
    order_sizes_series = pd.Series([10, 20, 5, 30, 8], index=idx)
    slippage2 = volume_model.calculate_slippage(test_data, 'BTCUSDT', order_size=order_sizes_series)
    print(f"\nVolume Based Slippage (order_size as Series):\n{slippage2*100}")
    
    # Slippage pour un ordre limite
    slippage_limit = volume_model.calculate_slippage(test_data, 'BTCUSDT', order_size=50, is_market_order=False)
    print(f"\nVolume Based Slippage (order_size=50, Limit Order):\n{slippage_limit*100}")
