# src/strategies/signal_formatter.py
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import pandas as pd
from decimal import Decimal
from loguru import logger

from src.core.constants import Trading
from src.core.exceptions import InvalidStrategyParamsError, SignalGenerationError


class SignalFormatter:
    """
    Formate et valide les signaux de trading selon le format standardisé.
    Assure la cohérence et la validité des signaux avant leur utilisation.
    """
    
    # Champs requis pour un signal
    REQUIRED_FIELDS = [
        'signal_type',
        'side',
        'pair',
        'entry_price_target'
    ]
    
    # Champs optionnels avec valeurs par défaut
    OPTIONAL_FIELDS = {
        'quantity': None,
        'confidence': 0.5,
        'metadata': {}
    }
    
    # Types de signaux valides
    VALID_SIGNAL_TYPES = [
        Trading.SIGNAL_TYPE_LONG,
        Trading.SIGNAL_TYPE_SHORT,
        Trading.SIGNAL_TYPE_NEUTRAL
    ]
    
    # Côtés valides
    VALID_SIDES = [
        Trading.SIDE_BUY,
        Trading.SIDE_SELL
    ]
    
    @staticmethod
    def format_signal(
        signal_data: Dict[str, Any],
        strategy_name: str,
        pair: str,
        timestamp: Optional[Union[datetime, pd.Timestamp]] = None
    ) -> Dict[str, Any]:
        """
        Formate un signal au format standardisé.
        
        Args:
            signal_data: Données brutes du signal
            strategy_name: Nom de la stratégie
            pair: Paire de trading
            timestamp: Timestamp du signal (défaut: maintenant)
            
        Returns:
            Signal formaté et validé
            
        Raises:
            SignalGenerationError: Si le signal est invalide
        """
        # Générer un ID unique pour le signal
        signal_id = str(uuid.uuid4())
        
        # Timestamp par défaut
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                
        # Construire le signal formaté
        formatted_signal = {
            'signal_id': signal_id,
            'timestamp': timestamp.isoformat(),
            'strategy_name': strategy_name,
            'pair': pair.upper()
        }
        
        # Ajouter les champs du signal
        formatted_signal.update(signal_data)
        
        # Ajouter les champs optionnels avec valeurs par défaut
        for field, default_value in SignalFormatter.OPTIONAL_FIELDS.items():
            if field not in formatted_signal:
                formatted_signal[field] = default_value
                
        # Valider le signal
        SignalFormatter._validate_signal(formatted_signal)
        
        # Formater l'ordre OCO si présent
        if 'oco_order' in formatted_signal:
            formatted_signal['oco_order'] = SignalFormatter._format_oco_order(
                formatted_signal['oco_order'],
                formatted_signal['signal_type'],
                formatted_signal['entry_price_target']
            )
            
        return formatted_signal
        
    @staticmethod
    def _validate_signal(signal: Dict[str, Any]) -> None:
        """
        Valide un signal formaté.
        
        Args:
            signal: Signal à valider
            
        Raises:
            SignalGenerationError: Si le signal est invalide
        """
        # Vérifier les champs requis
        for field in SignalFormatter.REQUIRED_FIELDS:
            if field not in signal:
                raise SignalGenerationError(
                    f"Missing required field '{field}' in signal"
                )
                
        # Valider le type de signal
        if signal['signal_type'] not in SignalFormatter.VALID_SIGNAL_TYPES:
            raise SignalGenerationError(
                f"Invalid signal type: {signal['signal_type']}. "
                f"Must be one of {SignalFormatter.VALID_SIGNAL_TYPES}"
            )
            
        # Valider le côté
        if signal['side'] not in SignalFormatter.VALID_SIDES:
            raise SignalGenerationError(
                f"Invalid side: {signal['side']}. "
                f"Must be one of {SignalFormatter.VALID_SIDES}"
            )
            
        # Valider la cohérence signal_type/side
        if signal['signal_type'] == Trading.SIGNAL_TYPE_LONG and signal['side'] != Trading.SIDE_BUY:
            raise SignalGenerationError(
                f"LONG signal must have side=BUY, got {signal['side']}"
            )
        elif signal['signal_type'] == Trading.SIGNAL_TYPE_SHORT and signal['side'] != Trading.SIDE_SELL:
            raise SignalGenerationError(
                f"SHORT signal must have side=SELL, got {signal['side']}"
            )
            
        # Valider les prix
        SignalFormatter._validate_prices(signal)
        
        # Valider la confidence
        confidence = signal.get('confidence', 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            raise SignalGenerationError(
                f"Invalid confidence value: {confidence}. Must be between 0 and 1"
            )
            
        # Valider la quantité si présente
        if 'quantity' in signal and signal['quantity'] is not None:
            if not isinstance(signal['quantity'], (int, float)) or signal['quantity'] <= 0:
                raise SignalGenerationError(
                    f"Invalid quantity: {signal['quantity']}. Must be positive"
                )
                
    @staticmethod
    def _validate_prices(signal: Dict[str, Any]) -> None:
        """
        Valide les prix dans un signal.
        
        Args:
            signal: Signal contenant les prix
            
        Raises:
            SignalGenerationError: Si les prix sont invalides
        """
        entry_price = signal.get('entry_price_target')
        
        # Valider le prix d'entrée
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            raise SignalGenerationError(
                f"Invalid entry price: {entry_price}. Must be positive"
            )
            
        # Si un ordre OCO est présent, valider ses prix
        if 'oco_order' in signal:
            oco = signal['oco_order']
            
            tp_price = oco.get('take_profit_price')
            sl_price = oco.get('stop_loss_price')
            
            if tp_price is not None:
                if not isinstance(tp_price, (int, float)) or tp_price <= 0:
                    raise SignalGenerationError(
                        f"Invalid take profit price: {tp_price}"
                    )
                    
            if sl_price is not None:
                if not isinstance(sl_price, (int, float)) or sl_price <= 0:
                    raise SignalGenerationError(
                        f"Invalid stop loss price: {sl_price}"
                    )
                    
            # Valider la cohérence des prix selon le type de signal
            if signal['signal_type'] == Trading.SIGNAL_TYPE_LONG:
                if tp_price and tp_price <= entry_price:
                    raise SignalGenerationError(
                        f"Take profit ({tp_price}) must be above entry ({entry_price}) for LONG"
                    )
                if sl_price and sl_price >= entry_price:
                    raise SignalGenerationError(
                        f"Stop loss ({sl_price}) must be below entry ({entry_price}) for LONG"
                    )
                    
            elif signal['signal_type'] == Trading.SIGNAL_TYPE_SHORT:
                if tp_price and tp_price >= entry_price:
                    raise SignalGenerationError(
                        f"Take profit ({tp_price}) must be below entry ({entry_price}) for SHORT"
                    )
                if sl_price and sl_price <= entry_price:
                    raise SignalGenerationError(
                        f"Stop loss ({sl_price}) must be above entry ({entry_price}) for SHORT"
                    )
                    
    @staticmethod
    def _format_oco_order(
        oco_data: Dict[str, Any],
        signal_type: str,
        entry_price: float
    ) -> Dict[str, Any]:
        """
        Formate les données d'un ordre OCO.
        
        Args:
            oco_data: Données brutes de l'ordre OCO
            signal_type: Type de signal (LONG/SHORT)
            entry_price: Prix d'entrée
            
        Returns:
            Ordre OCO formaté
        """
        formatted_oco = {}
        
        # Take Profit
        if 'take_profit_price' in oco_data or 'take_profit_pct' in oco_data:
            if 'take_profit_price' in oco_data:
                formatted_oco['take_profit_price'] = float(oco_data['take_profit_price'])
            elif 'take_profit_pct' in oco_data:
                tp_pct = float(oco_data['take_profit_pct'])
                if signal_type == Trading.SIGNAL_TYPE_LONG:
                    formatted_oco['take_profit_price'] = entry_price * (1 + tp_pct / 100)
                else:
                    formatted_oco['take_profit_price'] = entry_price * (1 - tp_pct / 100)
                    
        # Stop Loss
        if 'stop_loss_price' in oco_data or 'stop_loss_pct' in oco_data:
            if 'stop_loss_price' in oco_data:
                formatted_oco['stop_loss_price'] = float(oco_data['stop_loss_price'])
            elif 'stop_loss_pct' in oco_data:
                sl_pct = float(oco_data['stop_loss_pct'])
                if signal_type == Trading.SIGNAL_TYPE_LONG:
                    formatted_oco['stop_loss_price'] = entry_price * (1 - sl_pct / 100)
                else:
                    formatted_oco['stop_loss_price'] = entry_price * (1 + sl_pct / 100)
                    
        # Stop Limit Price (optionnel, pour les ordres stop limit)
        if 'stop_limit_price' in oco_data:
            formatted_oco['stop_limit_price'] = float(oco_data['stop_limit_price'])
        elif 'stop_loss_price' in formatted_oco:
            # Par défaut, légèrement en dessous/au-dessus du stop loss
            if signal_type == Trading.SIGNAL_TYPE_LONG:
                formatted_oco['stop_limit_price'] = formatted_oco['stop_loss_price'] * 0.999
            else:
                formatted_oco['stop_limit_price'] = formatted_oco['stop_loss_price'] * 1.001
                
        return formatted_oco
        
    @staticmethod
    def batch_format_signals(
        signals_data: List[Dict[str, Any]],
        strategy_name: str,
        pair: str,
        timestamp: Optional[Union[datetime, pd.Timestamp]] = None
    ) -> List[Dict[str, Any]]:
        """
        Formate un lot de signaux.
        
        Args:
            signals_data: Liste des données brutes de signaux
            strategy_name: Nom de la stratégie
            pair: Paire de trading
            timestamp: Timestamp commun (optionnel)
            
        Returns:
            Liste de signaux formatés
        """
        formatted_signals = []
        
        for signal_data in signals_data:
            try:
                formatted_signal = SignalFormatter.format_signal(
                    signal_data,
                    strategy_name,
                    pair,
                    timestamp
                )
                formatted_signals.append(formatted_signal)
                
            except SignalGenerationError as e:
                logger.warning(f"Skipping invalid signal: {e}")
                continue
                
        return formatted_signals
        
    @staticmethod
    def signal_to_order_params(
        signal: Dict[str, Any],
        quantity: float,
        order_type: str = Trading.ORDER_TYPE_LIMIT
    ) -> Dict[str, Any]:
        """
        Convertit un signal en paramètres d'ordre pour Binance.
        
        Args:
            signal: Signal formaté
            quantity: Quantité à trader
            order_type: Type d'ordre (LIMIT par défaut)
            
        Returns:
            Paramètres d'ordre pour l'API Binance
        """
        # Ordre d'entrée
        entry_order = {
            'symbol': signal['pair'],
            'side': signal['side'],
            'type': order_type,
            'quantity': quantity,
            'timeInForce': Trading.TIME_IN_FORCE_GTC
        }
        
        # Ajouter le prix pour les ordres LIMIT
        if order_type == Trading.ORDER_TYPE_LIMIT:
            entry_order['price'] = signal['entry_price_target']
            
        # Paramètres de l'ordre OCO si présent
        oco_params = None
        if 'oco_order' in signal:
            oco = signal['oco_order']
            
            # Déterminer le côté de l'ordre OCO (inverse du signal)
            oco_side = Trading.SIDE_SELL if signal['side'] == Trading.SIDE_BUY else Trading.SIDE_BUY
            
            oco_params = {
                'symbol': signal['pair'],
                'side': oco_side,
                'quantity': quantity,
                'price': oco.get('take_profit_price'),
                'stopPrice': oco.get('stop_loss_price'),
                'stopLimitPrice': oco.get('stop_limit_price'),
                'stopLimitTimeInForce': Trading.TIME_IN_FORCE_GTC
            }
            
        return {
            'entry_order': entry_order,
            'oco_order': oco_params
        }
        
    @staticmethod
    def filter_signals(
        signals: List[Dict[str, Any]],
        min_confidence: float = 0.5,
        signal_types: Optional[List[str]] = None,
        max_age_seconds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filtre une liste de signaux selon différents critères.
        
        Args:
            signals: Liste de signaux à filtrer
            min_confidence: Confidence minimale requise
            signal_types: Types de signaux à garder (None = tous)
            max_age_seconds: Âge maximum des signaux en secondes
            
        Returns:
            Liste de signaux filtrés
        """
        filtered = []
        now = datetime.now(timezone.utc)
        
        for signal in signals:
            # Filtrer par confidence
            if signal.get('confidence', 0) < min_confidence:
                continue
                
            # Filtrer par type
            if signal_types and signal['signal_type'] not in signal_types:
                continue
                
            # Filtrer par âge
            if max_age_seconds is not None:
                signal_time = datetime.fromisoformat(signal['timestamp'])
                if (now - signal_time).total_seconds() > max_age_seconds:
                    continue
                    
            filtered.append(signal)
            
        return filtered