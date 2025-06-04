# src/utils/exchange_utils.py
import asyncio
import math # Nécessaire pour math.log10 si utilisé, mais getcontext().prec est pour Decimal
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, getcontext, InvalidOperation, Context
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Literal, Any

from pydantic import BaseModel, Field, field_validator, model_validator 

from loguru import logger

# Configuration de la précision pour les calculs Decimal
# Une précision plus élevée peut être nécessaire pour certains calculs intermédiaires fins.
DECIMAL_CONTEXT = Context(prec=30) # Contexte Decimal avec une précision spécifique

# --- Pydantic Models ---

class BaseFilter(BaseModel):
    filterType: str

class PriceFilter(BaseModel): # Changé de BaseFilter pour éviter l'héritage inutile ici
    filterType: Literal["PRICE_FILTER"] = "PRICE_FILTER"
    minPrice: Decimal
    maxPrice: Decimal
    tickSize: Decimal

class LotSizeFilter(BaseModel):
    filterType: Literal["LOT_SIZE"] = "LOT_SIZE"
    minQty: Decimal
    maxQty: Decimal
    stepSize: Decimal

class MinNotionalFilter(BaseModel):
    filterType: Literal["MIN_NOTIONAL"] = "MIN_NOTIONAL"
    minNotional: Decimal
    applyToMarket: bool = True
    avgPriceMins: int = 5

class MarketLotSizeFilter(BaseModel):
    filterType: Literal["MARKET_LOT_SIZE"] = "MARKET_LOT_SIZE"
    minQty: Decimal
    maxQty: Decimal
    stepSize: Decimal

class SymbolInfo(BaseModel):
    symbol: str
    status: str
    baseAsset: str
    baseAssetPrecision: int
    quoteAsset: str
    quotePrecision: int
    quoteAssetPrecision: int 
    baseCommissionPrecision: int
    quoteCommissionPrecision: int
    orderTypes: List[str]
    icebergAllowed: bool
    ocoAllowed: bool
    quoteOrderQtyMarketAllowed: bool
    allowTrailingStop: bool
    cancelReplaceAllowed: bool
    isSpotTradingAllowed: bool
    isMarginTradingAllowed: bool
    filters: List[Dict[str, Any]] 
    permissions: List[str]

    # Attributs pour stocker les filtres parsés (non-Pydantic fields, initialisés dans model_validator)
    _price_filter_cache: Optional[PriceFilter] = None
    _lot_size_filter_cache: Optional[LotSizeFilter] = None
    _min_notional_filter_cache: Optional[MinNotionalFilter] = None
    _market_lot_size_filter_cache: Optional[MarketLotSizeFilter] = None

    @model_validator(mode='after')
    def init_cached_filters(self) -> 'SymbolInfo':
        for f_dict in self.filters:
            filter_type = f_dict.get("filterType")
            try:
                if filter_type == "PRICE_FILTER":
                    self._price_filter_cache = PriceFilter(**f_dict)
                elif filter_type == "LOT_SIZE":
                    self._lot_size_filter_cache = LotSizeFilter(**f_dict)
                elif filter_type == "MIN_NOTIONAL":
                    self._min_notional_filter_cache = MinNotionalFilter(**f_dict)
                elif filter_type == "MARKET_LOT_SIZE":
                    self._market_lot_size_filter_cache = MarketLotSizeFilter(**f_dict)
            except Exception as e:
                logger.warning(f"SymbolInfo: Échec du parsing du filtre {filter_type} pour {self.symbol}: {e}. Filtre ignoré.")
        return self

    @property
    def price_filter(self) -> Optional[PriceFilter]:
        return self._price_filter_cache

    @property
    def lot_size_filter(self) -> Optional[LotSizeFilter]:
        return self._lot_size_filter_cache

    @property
    def min_notional_filter(self) -> Optional[MinNotionalFilter]:
        return self._min_notional_filter_cache
    
    @property
    def market_lot_size_filter(self) -> Optional[MarketLotSizeFilter]:
        return self._market_lot_size_filter_cache

# --- Fonctions Utilitaires ---

def get_filter_value(symbol_info_model: SymbolInfo, filter_type: str, filter_key: str) -> Optional[Any]:
    if not isinstance(symbol_info_model, SymbolInfo):
        logger.warning(f"get_filter_value: symbol_info_model n'est pas une instance de SymbolInfo. Type: {type(symbol_info_model)}")
        return None

    filter_attr_map = {
        "PRICE_FILTER": "price_filter",
        "LOT_SIZE": "lot_size_filter",
        "MIN_NOTIONAL": "min_notional_filter",
        "MARKET_LOT_SIZE": "market_lot_size_filter"
    }
    
    model_filter_attr_name = filter_attr_map.get(filter_type.upper())
    if not model_filter_attr_name:
        logger.warning(f"get_filter_value: Type de filtre inconnu '{filter_type}'.")
        return None
        
    filter_instance = getattr(symbol_info_model, model_filter_attr_name, None)
    
    if filter_instance and hasattr(filter_instance, filter_key):
        return getattr(filter_instance, filter_key)
    
    logger.debug(f"get_filter_value: Filtre '{filter_type}' ou clé '{filter_key}' non trouvé pour {symbol_info_model.symbol}.")
    return None

def get_precision_from_filter(symbol_info_model: SymbolInfo, 
                              filter_type: Literal["PRICE_FILTER", "LOT_SIZE", "MARKET_LOT_SIZE"], 
                              key: Literal["tickSize", "stepSize"]) -> Optional[int]:
    if not isinstance(symbol_info_model, SymbolInfo):
        logger.warning(f"get_precision_from_filter: symbol_info_model n'est pas une instance de SymbolInfo.")
        return None

    size_value_decimal: Optional[Decimal] = None
    filter_instance: Optional[Union[PriceFilter, LotSizeFilter, MarketLotSizeFilter]] = None

    if filter_type == "PRICE_FILTER" and key == "tickSize":
        filter_instance = symbol_info_model.price_filter
        if filter_instance: size_value_decimal = filter_instance.tickSize
    elif filter_type == "LOT_SIZE" and key == "stepSize":
        filter_instance = symbol_info_model.lot_size_filter
        if filter_instance: size_value_decimal = filter_instance.stepSize
    elif filter_type == "MARKET_LOT_SIZE" and key == "stepSize":
        filter_instance = symbol_info_model.market_lot_size_filter
        if filter_instance: size_value_decimal = filter_instance.stepSize
    else:
        logger.warning(f"Combinaison filter_type/key non supportée: {filter_type}/{key} pour {symbol_info_model.symbol}")
        return None

    if size_value_decimal is None or size_value_decimal <= Decimal(0):
        logger.warning(f"Valeur de taille invalide ou non trouvée pour {filter_type}/{key} ({size_value_decimal}) pour {symbol_info_model.symbol}. Fallback à la précision de l'asset.")
        return symbol_info_model.quotePrecision if filter_type == "PRICE_FILTER" else symbol_info_model.baseAssetPrecision
        
    try:
        # Convertir en string pour compter les décimales de manière fiable
        # 'normalize()' retire les zéros de fin inutiles avant la conversion en tuple
        # Ex: Decimal('0.0100').normalize() -> Decimal('0.01')
        # Ex: Decimal('1.0').normalize() -> Decimal('1')
        exponent = size_value_decimal.normalize().as_tuple().exponent
        if isinstance(exponent, int): # L'exposant pour les nombres finis est un int
            return abs(exponent) # Le nombre de décimales est la valeur absolue de l'exposant négatif
        else: # Cas où l'exposant n'est pas un int (ex: pour NaN, Infini), ne devrait pas arriver pour tick/stepSize valides
             logger.error(f"Exposant inattendu pour {size_value_decimal}: {exponent}. Fallback.")
             return symbol_info_model.quotePrecision if filter_type == "PRICE_FILTER" else symbol_info_model.baseAssetPrecision
    except Exception as e:
        logger.error(f"Erreur lors de la détermination de la précision pour {size_value_decimal}: {e}")
        return symbol_info_model.quotePrecision if filter_type == "PRICE_FILTER" else symbol_info_model.baseAssetPrecision

def adjust_precision(
    value: Union[float, Decimal, str, int], 
    precision: int, 
    rounding_method: str = ROUND_DOWN 
) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        # Utiliser le contexte Decimal pour les opérations
        with DECIMAL_CONTEXT as ctx:
            d_value = ctx.create_decimal(str(value)) # Convertir en Decimal en utilisant le contexte

            if not isinstance(precision, int) or precision < 0:
                logger.warning(f"Précision invalide: {precision}. Utilisation de 8 par défaut.")
                precision = 8 

            # Créer le quantizer basé sur la précision
            # ex: pour precision=2, quantizer = Decimal('0.01')
            quantizer_str = '1e-' + str(precision)
            quantizer = ctx.create_decimal(quantizer_str)
            
            rounded_value = d_value.quantize(quantizer, rounding=rounding_method)
            return rounded_value
    except (TypeError, ValueError, InvalidOperation) as e:
        logger.error(f"Erreur lors de l'ajustement de la précision pour la valeur '{value}' à {precision} décimales: {e}")
        return None

async def format_price(price: Union[float, Decimal, str], symbol_info_model: SymbolInfo) -> Optional[str]:
    if not isinstance(symbol_info_model, SymbolInfo): 
        logger.warning("format_price: symbol_info_model n'est pas une instance de SymbolInfo. Retourne le prix tel quel.")
        return str(price)
    
    price_filter = symbol_info_model.price_filter
    if price_filter and price_filter.tickSize > Decimal(0):
        precision = get_precision_from_filter(symbol_info_model, "PRICE_FILTER", "tickSize")
        if precision is None: 
            precision = symbol_info_model.quotePrecision 
            logger.warning(f"format_price: Précision non déterminée pour {symbol_info_model.symbol} via tickSize, fallback à quotePrecision ({precision}).")

        tick_size = price_filter.tickSize
        d_price = Decimal(str(price))
        
        # (prix / tick_size) arrondi à 0 décimale (le plus proche), puis * tick_size
        adjusted_d_price_by_tick = (d_price / tick_size).quantize(Decimal('0'), rounding=ROUND_HALF_UP) * tick_size
        
        # Formater ensuite à la précision du tickSize (nombre de décimales)
        # La valeur est déjà un multiple de tickSize, il suffit de la formater en string.
        formatted_price_str = format(adjusted_d_price_by_tick, f'.{precision}f')
        return formatted_price_str
        
    logger.warning(f"format_price: Pas de filtre de prix valide pour {symbol_info_model.symbol}. Utilisation de quotePrecision.")
    return format(Decimal(str(price)), f'.{symbol_info_model.quotePrecision}f')

async def format_quantity(quantity: Union[float, Decimal, str], symbol_info_model: SymbolInfo, use_market_lot_size: bool = False) -> Optional[str]:
    if not isinstance(symbol_info_model, SymbolInfo): 
        logger.warning("format_quantity: symbol_info_model n'est pas une instance de SymbolInfo. Retourne la quantité telle quelle.")
        return str(quantity)

    lot_filter_to_use: Optional[Union[LotSizeFilter, MarketLotSizeFilter]] = None
    filter_name_for_log = ""

    if use_market_lot_size and symbol_info_model.market_lot_size_filter:
        lot_filter_to_use = symbol_info_model.market_lot_size_filter
        filter_name_for_log = "MARKET_LOT_SIZE"
    elif symbol_info_model.lot_size_filter:
        lot_filter_to_use = symbol_info_model.lot_size_filter
        filter_name_for_log = "LOT_SIZE"

    if lot_filter_to_use and lot_filter_to_use.stepSize > Decimal(0):
        precision = get_precision_from_filter(symbol_info_model, filter_name_for_log, "stepSize") # type: ignore
        if precision is None:
            precision = symbol_info_model.baseAssetPrecision
            logger.warning(f"format_quantity: Précision non déterminée pour {symbol_info_model.symbol} via {filter_name_for_log}, fallback à baseAssetPrecision ({precision}).")

        step_size = lot_filter_to_use.stepSize
        d_quantity = Decimal(str(quantity))

        if d_quantity < lot_filter_to_use.minQty:
             logger.warning(f"format_quantity: Quantité {d_quantity} < minQty {lot_filter_to_use.minQty} pour {symbol_info_model.symbol} ({filter_name_for_log}). La quantité ne sera pas ajustée à minQty ici, la validation d'ordre devrait le faire.")
             # Ne pas ajuster à minQty ici, la validation d'ordre est un meilleur endroit pour cette logique.

        # (quantité / step_size) arrondi vers le bas à 0 décimale, puis * step_size
        adjusted_d_quantity_by_step = (d_quantity / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
        
        # Formater ensuite à la précision du stepSize
        formatted_quantity_str = format(adjusted_d_quantity_by_step, f'.{precision}f')
        return formatted_quantity_str

    logger.warning(f"format_quantity: Pas de filtre de lot valide ({filter_name_for_log}) pour {symbol_info_model.symbol}. Utilisation de baseAssetPrecision.")
    return format(Decimal(str(quantity)), f'.{symbol_info_model.baseAssetPrecision}f')

# --- Mock client et fonction pour les tests ---
class MockBinanceClientForTests:
    _exchange_info_cache: Optional[Dict] = None
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        if self._exchange_info_cache is None:
            self._exchange_info_cache = { # Simule une partie de la réponse de get_exchange_info
                "symbols": [
                    {
                        "symbol": "BTCUSDC", "status": "TRADING", "baseAsset": "BTC", "baseAssetPrecision": 8,
                        "quoteAsset": "USDC", "quotePrecision": 2, "quoteAssetPrecision": 2, # quotePrecision pour USDC est souvent 2 ou 6
                        "baseCommissionPrecision": 8, "quoteCommissionPrecision": 8,
                        "orderTypes": ["LIMIT", "MARKET"], "icebergAllowed": False, "ocoAllowed": True,
                        "quoteOrderQtyMarketAllowed": True, "allowTrailingStop": False, "cancelReplaceAllowed": False,
                        "isSpotTradingAllowed": True, "isMarginTradingAllowed": False,
                        "filters": [
                            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000.00", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.00001", "maxQty": "100.0", "stepSize": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10.0", "applyToMarket": True, "avgPriceMins": 5}
                        ],
                        "permissions": ["SPOT"]
                    },
                    { # Symbole avec un stepSize différent pour la quantité
                        "symbol": "ETHUSDC", "status": "TRADING", "baseAsset": "ETH", "baseAssetPrecision": 8,
                        "quoteAsset": "USDC", "quotePrecision": 2, "quoteAssetPrecision": 2,
                        "baseCommissionPrecision": 8, "quoteCommissionPrecision": 8,
                        "orderTypes": ["LIMIT", "MARKET"], "icebergAllowed": False, "ocoAllowed": True,
                        "quoteOrderQtyMarketAllowed": True, "allowTrailingStop": False, "cancelReplaceAllowed": False,
                        "isSpotTradingAllowed": True, "isMarginTradingAllowed": False,
                        "filters": [
                            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "100000.00", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.0001", "maxQty": "1000.0", "stepSize": "0.001"}, # StepSize de 0.001 (3 décimales)
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10.0", "applyToMarket": True, "avgPriceMins": 5}
                        ],
                        "permissions": ["SPOT"]
                    }
                ]
            }
        
        for s_info in self._exchange_info_cache.get("symbols", []):
            if s_info.get("symbol") == symbol.upper():
                return s_info
        return None

mock_client_instance_for_tests = MockBinanceClientForTests()

async def get_symbol_info_model_for_tests(symbol: str) -> Optional[SymbolInfo]:
    raw_info = await mock_client_instance_for_tests.get_symbol_info(symbol.upper())
    if raw_info:
        try:
            return SymbolInfo(**raw_info)
        except Exception as e:
            logger.error(f"Test: Erreur de parsing SymbolInfo pour {symbol}: {e}. Données: {raw_info}")
            return None
    return None


if __name__ == '__main__':
    async def main_tests_exchange_utils():
        logger.remove() # Enlever les handlers par défaut pour ce test
        logger.add(lambda msg: print(msg, end=''), colorize=True, format="<level>{level[0]}</level>| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG")
        
        logger.info("--- Testing exchange_utils (adjust_precision focus) ---")
        
        btc_info_test: Optional[SymbolInfo] = await get_symbol_info_model_for_tests("BTCUSDC")
        eth_info_test: Optional[SymbolInfo] = await get_symbol_info_model_for_tests("ETHUSDC")

        if btc_info_test:
            logger.info(f"--- BTCUSDC (tickSize: {btc_info_test.price_filter.tickSize if btc_info_test.price_filter else 'N/A'}, stepSize: {btc_info_test.lot_size_filter.stepSize if btc_info_test.lot_size_filter else 'N/A'}) ---")
            price_prec_btc = get_precision_from_filter(btc_info_test, "PRICE_FILTER", "tickSize")
            qty_prec_btc = get_precision_from_filter(btc_info_test, "LOT_SIZE", "stepSize")
            logger.info(f"BTC Price Precision: {price_prec_btc}, Qty Precision: {qty_prec_btc}")

            test_price_btc = Decimal("45000.12876")
            adj_price_btc = adjust_precision(test_price_btc, price_prec_btc if price_prec_btc is not None else 2, ROUND_HALF_UP)
            logger.info(f"adjust_precision({test_price_btc}, {price_prec_btc}, ROUND_HALF_UP) -> {adj_price_btc}")
            fmt_price_btc = await format_price(float(test_price_btc), btc_info_test) # format_price attend un float
            logger.info(f"format_price({float(test_price_btc)}) -> {fmt_price_btc}")

            test_qty_btc = Decimal("0.12398765")
            adj_qty_btc = adjust_precision(test_qty_btc, qty_prec_btc if qty_prec_btc is not None else 8, ROUND_DOWN)
            logger.info(f"adjust_precision({test_qty_btc}, {qty_prec_btc}, ROUND_DOWN) -> {adj_qty_btc}")
            fmt_qty_btc = await format_quantity(float(test_qty_btc), btc_info_test)
            logger.info(f"format_quantity({float(test_qty_btc)}) -> {fmt_qty_btc}")
            
            # Test avec une quantité qui devrait être arrondie par stepSize
            # BTCUSDC stepSize = 0.00001 (5 décimales)
            test_qty_btc_2 = Decimal("0.123456") # Devrait devenir 0.12345
            fmt_qty_btc_2 = await format_quantity(float(test_qty_btc_2), btc_info_test)
            logger.info(f"format_quantity({float(test_qty_btc_2)}) -> {fmt_qty_btc_2} (expected 0.12345)")


        if eth_info_test:
            logger.info(f"--- ETHUSDC (tickSize: {eth_info_test.price_filter.tickSize if eth_info_test.price_filter else 'N/A'}, stepSize: {eth_info_test.lot_size_filter.stepSize if eth_info_test.lot_size_filter else 'N/A'}) ---")
            price_prec_eth = get_precision_from_filter(eth_info_test, "PRICE_FILTER", "tickSize")
            qty_prec_eth = get_precision_from_filter(eth_info_test, "LOT_SIZE", "stepSize")
            logger.info(f"ETH Price Precision: {price_prec_eth}, Qty Precision: {qty_prec_eth}")

            test_price_eth = Decimal("3000.755")
            adj_price_eth = adjust_precision(test_price_eth, price_prec_eth if price_prec_eth is not None else 2, ROUND_HALF_UP)
            logger.info(f"adjust_precision({test_price_eth}, {price_prec_eth}, ROUND_HALF_UP) -> {adj_price_eth}")
            fmt_price_eth = await format_price(float(test_price_eth), eth_info_test)
            logger.info(f"format_price({float(test_price_eth)}) -> {fmt_price_eth}")


            # ETHUSDC stepSize = 0.001 (3 décimales)
            test_qty_eth = Decimal("1.234567") # Devrait devenir 1.234
            adj_qty_eth = adjust_precision(test_qty_eth, qty_prec_eth if qty_prec_eth is not None else 8, ROUND_DOWN)
            logger.info(f"adjust_precision({test_qty_eth}, {qty_prec_eth}, ROUND_DOWN) -> {adj_qty_eth}")
            fmt_qty_eth = await format_quantity(float(test_qty_eth), eth_info_test)
            logger.info(f"format_quantity({float(test_qty_eth)}) -> {fmt_qty_eth} (expected 1.234)")

            test_qty_eth_2 = Decimal("1.9999") # Devrait devenir 1.999
            fmt_qty_eth_2 = await format_quantity(float(test_qty_eth_2), eth_info_test)
            logger.info(f"format_quantity({float(test_qty_eth_2)}) -> {fmt_qty_eth_2} (expected 1.999)")
            
            test_qty_eth_3 = Decimal("0.0005") # Devrait devenir 0.000 car < minQty et < stepSize
            # minQty pour ETHUSDC est 0.0001, stepSize est 0.001.
            # format_quantity arrondit à stepSize, donc 0.0005 / 0.001 -> 0 * 0.001 = 0.000
            fmt_qty_eth_3 = await format_quantity(float(test_qty_eth_3), eth_info_test)
            logger.info(f"format_quantity({float(test_qty_eth_3)}) -> {fmt_qty_eth_3} (expected 0.000)")


    asyncio.run(main_tests_exchange_utils())
