# src/optimization/wfo_engine.py

"""
Ce module fournit un moteur avancé pour la génération de découpages (splits)
dans le cadre d'une Walk-Forward Optimization (WFO) ou d'une validation croisée.

Il remplace le générateur de splits basique par un `WFOSplitGenerator` robuste,
capable d'appliquer des techniques financières essentielles comme le purging et
l'embargo, et supportant plusieurs stratégies de découpage (temporelle, adaptative).
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterator

# --- Imports Corrigés ---
# Les imports sont maintenant absolus depuis la racine `src` pour éviter les ModuleNotFoundError.
from src.optimization.config import WFOConfig, FixedSplitConfig, AdaptiveVolatilitySplitConfig
from src.core.exceptions import ConfigurationError, DataError

logger = logging.getLogger(__name__)


class WFOSplitGenerator:
    """
    Génère des découpages (splits) In-Sample (IS) et Out-of-Sample (OOS)
    pour la Walk-Forward Optimization, en appliquant les meilleures pratiques
    de Machine Learning Financier.
    """

    def __init__(self, config: WFOConfig):
        """
        Initialise le générateur de splits avec la configuration WFO.

        Args:
            config: L'objet de configuration Pydantic `WFOConfig`.
        """
        self.config = config
        logger.info(f"WFOSplitGenerator initialisé avec la méthode de découpage: {config.split_config.type}")

    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Méthode principale pour générer la liste des découpages.

        Args:
            data: DataFrame contenant les données de marché, doit avoir un DatetimeIndex.

        Returns:
            Une liste de tuples, où chaque tuple contient deux DatetimeIndex:
            (index_in_sample, index_out_of_sample).

        Raises:
            DataError: Si les données ne sont pas au format attendu.
            ConfigurationError: Si la méthode de découpage n'est pas supportée.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataError("Les données doivent avoir un DatetimeIndex pour le découpage temporel.")

        split_config = self.config.split_config
        
        if isinstance(split_config, FixedSplitConfig):
            return list(self._split_fixed_time(data, split_config))
        elif isinstance(split_config, AdaptiveVolatilitySplitConfig):
            return list(self._split_adaptive_volatility(data, split_config))
        else:
            raise ConfigurationError(f"Type de découpage non supporté: {type(split_config)}")

    def _calculate_is_days(self, total_days: int, config: FixedSplitConfig) -> int:
        """
        Calcule la durée de la période In-Sample si elle n'est pas fournie.
        """
        # La durée totale consommée par les périodes OOS, que la fenêtre soit glissante ou expansive.
        total_oos_days = config.n_splits * config.oos_days
        
        is_days = total_days - total_oos_days
        
        logger.info(f"Le paramètre 'is_days' n'a pas été fourni. Calcul automatique :")
        logger.info(f"  Durée totale des données : {total_days} jours")
        logger.info(f"  Durée totale des OOS ({config.n_splits} splits * {config.oos_days} jours) : {total_oos_days} jours")
        logger.info(f"  Durée calculée pour IS : {is_days} jours")

        if is_days <= 0:
            raise ConfigurationError(
                f"La durée totale des données ({total_days} jours) est insuffisante "
                f"pour le nombre de splits ({config.n_splits}) et la durée OOS ({config.oos_days} jours). "
                "Réduisez 'n_splits' ou 'oos_days'."
            )
        return is_days


    def _split_fixed_time(self, data: pd.DataFrame, config: FixedSplitConfig) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Génère des splits basés sur des durées temporelles fixes.
        Calcule dynamiquement `is_days` si non fourni.
        """
        logger.info("Génération de splits avec la méthode 'FIXED_TIME'.")
        
        total_days = (data.index[-1] - data.index[0]).days
        
        # --- Logique de calcul dynamique ---
        is_days = config.is_days
        if is_days is None:
            is_days = self._calculate_is_days(total_days, config)
        # ------------------------------------

        is_duration = pd.Timedelta(days=is_days)
        oos_duration = pd.Timedelta(days=config.oos_days)
        
        if config.expanding_window:
            logger.info("Configuration en mode 'fenêtre expansive'.")
            initial_is_end = data.index[0] + is_duration
            split_step = oos_duration
            
            for i in range(config.n_splits):
                is_start = data.index[0]
                is_end = initial_is_end + i * split_step
                
                oos_start = is_end
                oos_end = oos_start + oos_duration

                if oos_end > data.index[-1]:
                    logger.warning(f"Arrêt prématuré à {i+1}/{config.n_splits} splits car la fin des données a été atteinte.")
                    break

                yield self._apply_purge_and_embargo(data.index, is_start, is_end, oos_start, oos_end)

        else: # Fenêtre glissante
            logger.info("Configuration en mode 'fenêtre glissante'.")
            split_step = oos_duration
            
            for i in range(config.n_splits):
                is_start = data.index[0] + i * split_step
                is_end = is_start + is_duration
                
                oos_start = is_end
                oos_end = oos_start + oos_duration

                if oos_end > data.index[-1]:
                    logger.warning(f"Arrêt prématuré à {i+1}/{config.n_splits} splits car la fin des données a été atteinte.")
                    break
                
                yield self._apply_purge_and_embargo(data.index, is_start, is_end, oos_start, oos_end)

    def _split_adaptive_volatility(self, data: pd.DataFrame, config: AdaptiveVolatilitySplitConfig) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Génère des splits basés sur la volatilité, pour des périodes plus homogènes.
        L'idée est de découper les données lorsque la volatilité cumulative atteint un seuil.
        """
        logger.info("Génération de splits avec la méthode 'ADAPTIVE_VOLATILITY'.")

        if 'close' not in data.columns:
            raise DataError("La colonne 'close' est requise pour le découpage basé sur la volatilité.")

        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=config.volatility_window).std().dropna() * np.sqrt(252) # Volatilité annualisée
        
        # Le seuil est basé sur la volatilité moyenne de la période
        target_volatility_sum = volatility.mean() * config.target_volatility_multiplier

        splits = []
        current_start_idx = 0
        
        while current_start_idx < len(data) - 1:
            cumulative_vol = 0
            split_point_idx = -1

            # Trouver le prochain point de découpage
            for i in range(current_start_idx + 1, len(data)):
                date = data.index[i]
                if date in volatility.index:
                    cumulative_vol += volatility.loc[date]
                
                is_duration_days = (date - data.index[current_start_idx]).days
                if cumulative_vol >= target_volatility_sum and is_duration_days >= config.min_is_days:
                    split_point_idx = i
                    break
            
            if split_point_idx == -1: # Fin des données atteinte
                break
                
            is_start = data.index[current_start_idx]
            is_end = data.index[split_point_idx]
            
            oos_end_date = is_end + pd.Timedelta(days=config.min_oos_days)
            try:
                # Utiliser 'bfill' pour prendre la prochaine date disponible si la date exacte n'existe pas
                oos_end_idx = data.index.get_loc(oos_end_date, method='bfill')
            except KeyError:
                # Si même bfill ne trouve rien, c'est que nous sommes à la fin
                break


            if oos_end_idx >= len(data):
                break

            oos_start = is_end
            oos_end = data.index[oos_end_idx]

            yield self._apply_purge_and_embargo(data.index, is_start, is_end, oos_start, oos_end)

            current_start_idx = split_point_idx # Le prochain split commence ici

    def _apply_purge_and_embargo(self,
                                 full_index: pd.DatetimeIndex,
                                 is_start: pd.Timestamp,
                                 is_end: pd.Timestamp,
                                 oos_start: pd.Timestamp,
                                 oos_end: pd.Timestamp
                                 ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Applique le purging et l'embargo à un couple de périodes IS/OOS.
        """
        purging_duration = pd.Timedelta(days=self.config.purge_and_embargo.purging_days)
        embargo_duration = pd.Timedelta(days=self.config.purge_and_embargo.embargo_days)
        
        # Appliquer le purging à la fin de la période IS
        purged_is_end = is_end - purging_duration
        
        # Appliquer l'embargo au début de la période OOS
        embargoed_oos_start = oos_start + embargo_duration

        # Sélectionner les indices correspondants
        is_indices = full_index[(full_index >= is_start) & (full_index < purged_is_end)]
        oos_indices = full_index[(full_index >= embargoed_oos_start) & (full_index < oos_end)]

        # Vérifier que les splits ne sont pas vides après purge/embargo
        if is_indices.empty or oos_indices.empty:
            logger.warning("Un split est vide après application du purging/embargo. "
                           "Vérifiez que les durées (IS/OOS) sont suffisamment longues.")
            # Retourner des index vides pour que ce split soit ignoré
            return pd.DatetimeIndex([]), pd.DatetimeIndex([])

        logger.debug(f"Split generated: "
                     f"IS: {is_indices.min()} -> {is_indices.max()} ({len(is_indices)} rows), "
                     f"OOS: {oos_indices.min()} -> {oos_indices.max()} ({len(oos_indices)} rows)")
                     
        return is_indices, oos_indices


# Note : Le bloc ci-dessous est pour la démonstration.
# L'exécuter directement (`python src/optimization/wfo_engine.py`) échouera
# car Python ne trouvera pas les modules dans `src` sans configuration
# supplémentaire du chemin (sys.path), ce qui est normal pour un module
# faisant partie d'un package plus large.
if __name__ == '__main__':
    print("--- Démonstration de WFOSplitGenerator ---")
    
    # Créer un faux jeu de données
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', end='2024-12-31', freq='D'))
    mock_data = pd.DataFrame(
        {'close': np.random.randn(len(dates)).cumsum() + 100},
        index=dates
    )

    print("\n--- Test avec Fenêtre Glissante et Calcul Automatique de 'is_days' ---")
    fixed_config_dict = {
        "enabled": True,
        "split_config": {
            "type": "FIXED_TIME",
            "is_days": None, # Non spécifié pour le calcul automatique
            "oos_days": 60,
            "n_splits": 10,
            "expanding_window": False
        },
        "purge_and_embargo": {"purging_days": 7, "embargo_days": 3},
        "validation_config": {"n_best_trials_to_validate": 1}
    }
    
    # NOTE: Pour exécuter ce bloc, il faudrait pouvoir importer WFOConfig.
    # Dans un projet structuré, ce n'est pas possible directement.
    # Les lignes ci-dessous sont donc commentées mais illustrent l'usage.
    
    # wfo_config_fixed = WFOConfig.parse_obj(fixed_config_dict)
    # splitter_fixed = WFOSplitGenerator(wfo_config_fixed)
    
    # try:
    #     splits = splitter_fixed.split(mock_data)
    #     print(f"Nombre de splits générés: {len(splits)}")
    #     if splits:
    #         is_idx, oos_idx = splits[0]
    #         print(f"Premier split IS: {is_idx[0]} -> {is_idx[-1]} (taille: {len(is_idx)})")
    #         print(f"Premier split OOS: {oos_idx[0]} -> {oos_idx[-1]} (taille: {len(oos_idx)})")
    #         # Vérification du gap
    #         gap = oos_idx[0] - is_idx[-1]
    #         print(f"Gap total (purge + embargo): {gap.days} jours")
    # except ConfigurationError as e:
    #     print(f"Erreur de configuration capturée : {e}")
    
    print("\nLe code de démonstration est commenté car il ne peut pas être exécuté directement.")
    print("Veuillez utiliser ce module en l'important dans d'autres parties de votre application.")
