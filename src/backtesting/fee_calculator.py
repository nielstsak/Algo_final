# src/backtesting/fee_calculator.py
import pandas as pd
from typing import Union, Optional
from loguru import logger

class FeeCalculator:
    """
    Classe de base pour le calcul des frais de transaction.
    """
    def calculate_fees(
        self,
        data: pd.DataFrame, # Données OHLCV ou de prix
        symbol: str,
        is_maker: Union[bool, pd.Series] = False, # Si True, frais maker, sinon taker
        trade_value: Optional[Union[float, pd.Series]] = None # Valeur du trade si connue
    ) -> Union[float, pd.Series]:
        """
        Calcule les frais pour une transaction.
        Doit être surchargée par les classes enfants.

        Args:
            data: DataFrame contenant les données de prix ou OHLCV.
                  Utilisé si les frais dépendent du prix ou de la volatilité.
            symbol: Symbole de la paire de trading.
            is_maker: Si True, applique les frais "maker", sinon "taker".
                      Peut être une série booléenne alignée avec les trades.
            trade_value: La valeur du trade. Si fournie, les frais peuvent être calculés
                         directement comme un pourcentage de cette valeur. Si None,
                         les frais retournés sont un taux (ex: 0.001 pour 0.1%).

        Returns:
            Taux de frais (ex: 0.001) ou montant des frais si trade_value est fourni.
            Peut être un float (frais fixes) ou une pd.Series (frais variables).
        """
        raise NotImplementedError("La méthode calculate_fees doit être implémentée par la sous-classe.")

class BinanceFeeCalculator(FeeCalculator):
    """
    Calcule les frais de transaction pour Binance.
    Ceci est une implémentation simplifiée. Les frais réels peuvent dépendre
    du niveau VIP, de l'utilisation de BNB pour payer les frais, etc.
    """
    # Taux de frais standards (peuvent être ajustés)
    # Spot Trading
    SPOT_TAKER_FEE_DEFAULT = 0.001  # 0.1%
    SPOT_MAKER_FEE_DEFAULT = 0.001  # 0.1%
    SPOT_TAKER_FEE_BNB_DISCOUNT = 0.00075 # 0.075% (avec réduction BNB de 25%)
    SPOT_MAKER_FEE_BNB_DISCOUNT = 0.00075 # 0.075%

    # Futures Trading (USDⓈ-M) - Exemple, vérifiez les taux actuels
    FUTURES_TAKER_FEE_DEFAULT = 0.0004  # 0.04%
    FUTURES_MAKER_FEE_DEFAULT = 0.0002  # 0.02%
    # Les réductions BNB pour les futures sont différentes

    def __init__(
        self,
        use_bnb_discount_spot: bool = False, # Si True et applicable, utilise les taux réduits BNB pour spot
        is_futures: bool = False, # Si True, utilise les taux futures
        custom_taker_fee: Optional[float] = None,
        custom_maker_fee: Optional[float] = None
    ):
        self.use_bnb_discount_spot = use_bnb_discount_spot
        self.is_futures = is_futures

        if custom_taker_fee is not None and custom_maker_fee is not None:
            self.taker_fee = custom_taker_fee
            self.maker_fee = custom_maker_fee
            logger.info(f"BinanceFeeCalculator: Utilisation de frais personnalisés - Taker: {self.taker_fee*100:.4f}%, Maker: {self.maker_fee*100:.4f}%")
        elif is_futures:
            self.taker_fee = self.FUTURES_TAKER_FEE_DEFAULT
            self.maker_fee = self.FUTURES_MAKER_FEE_DEFAULT
            logger.info(f"BinanceFeeCalculator (Futures): Taker: {self.taker_fee*100:.4f}%, Maker: {self.maker_fee*100:.4f}%")
        else: # Spot
            if use_bnb_discount_spot:
                self.taker_fee = self.SPOT_TAKER_FEE_BNB_DISCOUNT
                self.maker_fee = self.SPOT_MAKER_FEE_BNB_DISCOUNT
                logger.info(f"BinanceFeeCalculator (Spot BNB Discount): Taker: {self.taker_fee*100:.4f}%, Maker: {self.maker_fee*100:.4f}%")
            else:
                self.taker_fee = self.SPOT_TAKER_FEE_DEFAULT
                self.maker_fee = self.SPOT_MAKER_FEE_DEFAULT
                logger.info(f"BinanceFeeCalculator (Spot Standard): Taker: {self.taker_fee*100:.4f}%, Maker: {self.maker_fee*100:.4f}%")


    def calculate_fees(
        self,
        data: pd.DataFrame,
        symbol: str,
        is_maker: Union[bool, pd.Series] = False,
        trade_value: Optional[Union[float, pd.Series]] = None
    ) -> Union[float, pd.Series]:
        """
        Calcule les frais de transaction pour Binance.

        Args:
            data: DataFrame (non utilisé dans cette implémentation simple, mais gardé pour compatibilité).
            symbol: Symbole de la paire (non utilisé ici, mais pourrait l'être pour des frais spécifiques par paire).
            is_maker: Si True, applique les frais "maker", sinon "taker".
                      Peut être une série booléenne.
            trade_value: Non utilisé par vectorbt pour les taux de frais. Vectorbt attend un taux.

        Returns:
            Taux de frais (ex: 0.001 pour 0.1%).
            Si `is_maker` est une Series, retourne une Series de taux.
        """
        if isinstance(is_maker, pd.Series):
            # Si is_maker est une série (par exemple, pour chaque trade),
            # retourner une série de frais correspondante.
            fees_series = pd.Series(index=is_maker.index, dtype=float)
            fees_series[is_maker] = self.maker_fee
            fees_series[~is_maker] = self.taker_fee
            return fees_series
        else:
            # Si is_maker est un simple booléen (frais fixes pour tous les trades)
            return self.maker_fee if is_maker else self.taker_fee

# Exemple d'utilisation:
if __name__ == "__main__":
    # Spot standard
    calc_spot = BinanceFeeCalculator()
    print(f"Spot Taker Fee: {calc_spot.calculate_fees(None, 'BTCUSDT', is_maker=False)}") # type: ignore
    print(f"Spot Maker Fee: {calc_spot.calculate_fees(None, 'BTCUSDT', is_maker=True)}") # type: ignore

    # Spot avec réduction BNB
    calc_spot_bnb = BinanceFeeCalculator(use_bnb_discount_spot=True)
    print(f"Spot BNB Taker Fee: {calc_spot_bnb.calculate_fees(None, 'BTCUSDT', is_maker=False)}") # type: ignore
    print(f"Spot BNB Maker Fee: {calc_spot_bnb.calculate_fees(None, 'BTCUSDT', is_maker=True)}") # type: ignore

    # Futures
    calc_futures = BinanceFeeCalculator(is_futures=True)
    print(f"Futures Taker Fee: {calc_futures.calculate_fees(None, 'BTCUSDT', is_maker=False)}") # type: ignore
    print(f"Futures Maker Fee: {calc_futures.calculate_fees(None, 'BTCUSDT', is_maker=True)}") # type: ignore

    # Frais personnalisés
    calc_custom = BinanceFeeCalculator(custom_taker_fee=0.0005, custom_maker_fee=0.0001) # type: ignore
    print(f"Custom Taker Fee: {calc_custom.calculate_fees(None, 'BTCUSDT', is_maker=False)}") # type: ignore
    print(f"Custom Maker Fee: {calc_custom.calculate_fees(None, 'BTCUSDT', is_maker=True)}") # type: ignore

    # Avec une série pour is_maker
    index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    is_maker_series = pd.Series([False, True, False], index=index)
    fees_s = calc_spot.calculate_fees(None, 'BTCUSDT', is_maker=is_maker_series) # type: ignore
    print(f"\nFees Series:\n{fees_s}")
