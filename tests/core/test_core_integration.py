# tests/core/test_core_integration.py
import pytest
from pathlib import Path
import yaml
from loguru import logger # Importer directement pour émettre des logs

# Importations des modules du projet
from src.core.config import Settings, load_settings, CONFIG_FILE_PATH
# setup_logging est appelé à la fin de src.core.config lors de l'import de settings
# La ligne incorrecte "from src.core.constants import AppConfig as AppConstantsConfig" a été supprimée.
from src.core.exceptions import AlgoBotException, ConfigurationError, DataError # Exemples d'exceptions

# --- Fixtures pour le test d'intégration ---

@pytest.fixture(scope="module")
def temp_config_for_integration(tmp_path_factory) -> Path:
    """
    Crée un fichier de configuration temporaire minimal pour les tests d'intégration.
    Ce fichier sera utilisé pour initialiser l'objet `settings`.
    """
    config_content = {
        "app": {
            "name": "IntegrationTestBot",
            "version": "0.0.1",
            "environment": "test"
        },
        "binance": { # Requis par BinanceConfigModel dans Settings
            # Les clés API et Secret doivent venir de l'environnement pour ce test
        },
        "database": { # Requis par DatabaseConfigModel
            "db_type": "sqlite",
            "sqlite_path": ":memory:" # Simple pour les tests
        },
        "data": { # Requis par DataConfig
            "storage_path": str(tmp_path_factory.mktemp("data_integration") / "storage"),
            # Les autres champs de DataConfig utiliseront leurs valeurs par défaut
        },
        "trading": { # Requis par TradingConfig (validé dans Settings)
            "base_currency": "USDT",
            "allowed_pairs": ["BTCUSDT", "ETHUSDT"]
        },
        "monitoring": { # Pour configurer le logging via ce fichier
            "log_level_console": "DEBUG",
            "log_to_file_enabled": False # Désactiver les logs fichiers pour ce test simple
        }
        # Les sections risk et redis utiliseront leurs valeurs par défaut ou seront None
    }
    config_dir = tmp_path_factory.mktemp("config_integration")
    config_file = config_dir / "config_integration.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    return config_file

@pytest.fixture(scope="module")
def integrated_settings(temp_config_for_integration: Path, monkeypatch) -> Settings: # Correction: monkeypatch_module -> monkeypatch
    """
    Charge les settings en utilisant le fichier de configuration temporaire
    et des variables d'environnement mockées pour les secrets.
    Utilise monkeypatch (avec sa portée par défaut ou celle définie par son utilisation ici dans une fixture module)
    pour que CONFIG_FILE_PATH soit patché pour toute la durée du module.
    """
    monkeypatch.setattr("src.core.config.CONFIG_FILE_PATH", temp_config_for_integration)
    
    # Mocker les variables d'environnement minimales requises par BinanceConfigModel
    monkeypatch.setenv("BINANCE_API_KEY", "test_integration_api_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_integration_api_secret")

    # Recharger le module settings pour qu'il utilise le CONFIG_FILE_PATH patché
    # et les variables d'environnement.
    import importlib
    import src.core.config # Importe le module pour pouvoir le recharger
    importlib.reload(src.core.config) # Recharge le module config
    
    # `settings` sera maintenant l'instance rechargée depuis `src.core.config`
    # qui aura utilisé le CONFIG_FILE_PATH patché et les variables d'environnement.
    return src.core.config.settings


# --- Test d'Intégration ---

def test_core_foundations_integration(integrated_settings: Settings, caplog):
    """
    Teste l'intégration basique des modules core:
    1. Chargement de la configuration (implicite par la fixture `integrated_settings`).
    2. Initialisation du logging (implicite par le chargement de `settings` si `config.py` est modifié).
    3. Émission de logs.
    4. Accès aux constantes.
    5. Levée et capture d'une exception personnalisée.
    """
    # 1. & 2. Configuration et Logging (implicitement initialisés par la fixture)
    # Le module `src.core.config` devrait avoir appelé `setup_logging(settings)` à la fin.
    # On vérifie que l'instance `settings` est bien celle attendue.
    assert integrated_settings.app.name == "IntegrationTestBot"
    logger.info(f"Test d'intégration : Configuration '{integrated_settings.app.name}' chargée et logging initialisé.")
    logger.debug(f"Test d'intégration : Environnement '{integrated_settings.app.environment}'.")

    # 3. Émettre quelques messages de log à différents niveaux
    logger.info("Ceci est un message INFO du test d'intégration.")
    logger.warning("Ceci est un message WARNING du test d'intégration.")
    logger.debug("Ceci est un message DEBUG du test d'intégration. Devrait être visible si log_level_console est DEBUG.")

    # 4. Accéder à une ou plusieurs constantes depuis src.core.constants
    from src.core.constants import Trading, System # Importation correcte des constantes
    
    assert Trading.SIDE_BUY == "BUY"
    logger.info(f"Test d'intégration : Constante Trading.SIDE_BUY = '{Trading.SIDE_BUY}' accessible.")
    assert System.MAX_KLINES_PER_BINANCE_REQUEST > 0
    logger.info(f"Test d'intégration : Constante System.MAX_KLINES_PER_BINANCE_REQUEST = '{System.MAX_KLINES_PER_BINANCE_REQUEST}' accessible.")


    # 5. Simuler une condition d'erreur attendue pour lever une des exceptions personnalisées
    error_message = "Erreur de données simulée pour le test d'intégration."
    error_context = {"pair": "BTCUSDT", "reason": "simulation"}

    with pytest.raises(DataError) as exc_info:
        # Simuler une fonction qui pourrait lever DataError
        def simulate_data_issue():
            raise DataError(error_message, original_exception=ValueError("Cause simulée"), **error_context)
        simulate_data_issue()

    # Capturer cette exception et vérifier son type et son message
    assert isinstance(exc_info.value, DataError)
    assert isinstance(exc_info.value, AlgoBotException) # Vérifier l'héritage
    assert error_message in str(exc_info.value)
    assert "ValueError: Cause simulée" in str(exc_info.value) # Vérifier l'exception originale
    assert exc_info.value.context["pair"] == "BTCUSDT"
    assert exc_info.value.context["reason"] == "simulation"
    logger.error(f"Test d'intégration : Exception personnalisée DataError levée et capturée comme prévu : {exc_info.value}")

    # 6. Vérifier que les logs émis pendant ce test apparaissent correctement configurés
    # `caplog` est une fixture pytest qui capture les logs émis.
    
    log_records_texts = [record.message for record in caplog.records]
    log_records_levels = [record.levelname for record in caplog.records]

    assert "Ceci est un message INFO du test d'intégration." in log_records_texts
    assert "Ceci est un message WARNING du test d'intégration." in log_records_texts
    assert "Ceci est un message DEBUG du test d'intégration." in log_records_texts
    
    # Vérifier les niveaux
    try:
        info_idx = log_records_texts.index("Ceci est un message INFO du test d'intégration.")
        assert log_records_levels[info_idx] == "INFO"
    except ValueError:
        pytest.fail("Message INFO non trouvé dans les logs capturés.")

    try:
        debug_idx = log_records_texts.index("Ceci est un message DEBUG du test d'intégration.")
        assert log_records_levels[debug_idx] == "DEBUG"
    except ValueError:
        pytest.fail("Message DEBUG non trouvé dans les logs capturés.")
        
    logger.success("Test d'intégration des fondations core terminé avec succès.")

