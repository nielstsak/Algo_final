# src/core/logging_config.py
import sys
import os
from pathlib import Path
from loguru import logger
from typing import TYPE_CHECKING, Optional
import requests # Pour le handler de webhook d'alerte

if TYPE_CHECKING:
    from src.core.config import Settings, MonitoringConfig # Prévention de l'import circulaire au runtime

# Détermine la racine du projet. Ajustez si la structure de votre projet est différente.
# src/core/logging_config.py -> src/core -> src -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

CONSOLE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

class AlertWebhookHandler:
    """
    Un handler Loguru personnalisé pour envoyer des messages à un webhook.
    """
    def __init__(self, webhook_url: str, min_level: str = "ERROR"):
        self.webhook_url = webhook_url
        self.min_level_value = logger.level(min_level.upper()).no

    def write(self, message):
        record = message.record
        if record["level"].no >= self.min_level_value:
            try:
                payload = {
                    "text": f"[{record['level'].name}] {record['time'].strftime('%Y-%m-%d %H:%M:%S')} | {record['name']}:{record['function']}:{record['line']} - {record['message']}"
                    # Adaptez le payload au format attendu par votre webhook (Slack, Discord, etc.)
                }
                # Pour Slack, le payload pourrait être : {"text": message_content}
                # Pour Discord, il pourrait être : {"content": message_content}
                response = requests.post(self.webhook_url, json=payload, timeout=5)
                response.raise_for_status()
            except requests.RequestException as e:
                # Que faire si l'envoi au webhook échoue ? Logger localement.
                # Éviter une boucle infinie de logging.
                sys.stderr.write(f"Failed to send log to webhook: {e}\nOriginal message: {record['message']}\n")
            except Exception as e:
                sys.stderr.write(f"Unexpected error in AlertWebhookHandler: {e}\nOriginal message: {record['message']}\n")


def setup_logging(config: 'Settings'):
    """
    Configure le système de logging Loguru basé sur les paramètres fournis.
    """
    monitoring_config: 'MonitoringConfig' = config.monitoring
    
    logger.remove()  # Retirer tous les handlers précédents pour éviter les duplications

    # 1. Handler Console
    logger.add(
        sys.stderr,
        level=monitoring_config.log_level_console.upper(),
        format=CONSOLE_LOG_FORMAT,
        colorize=True,
        enqueue=True, # Pour logging asynchrone et non bloquant
        backtrace=True, # Améliore les tracebacks
        diagnose=True   # Améliore les diagnostics d'erreurs
    )

    # 2. Handler Fichier (JSON structuré)
    if monitoring_config.log_to_file_enabled:
        LOGS_DIR.mkdir(parents=True, exist_ok=True) # S'assurer que le dossier de logs existe
        
        # Le nom du fichier est dynamique grâce à {time} dans le path
        file_log_path_template = LOGS_DIR / "app_{time:YYYY-MM-DD}.log"
        
        logger.add(
            file_log_path_template,
            level=monitoring_config.log_level_file.upper(),
            rotation=monitoring_config.log_rotation,
            retention=monitoring_config.log_retention,
            compression=monitoring_config.log_compression,
            serialize=True,  # IMPORTANT pour les logs JSON structurés
            enqueue=True,    # Pour logging asynchrone et non bloquant
            backtrace=True,  # Améliore les tracebacks pour les erreurs
            diagnose=True    # Améliore les diagnostics d'erreurs pour le contexte
        )

    # 3. (Optionnel) Handler pour Alertes Externes via Webhook
    if monitoring_config.alert_webhook_url:
        webhook_url_str = str(monitoring_config.alert_webhook_url) # Pydantic HttpUrl to str
        alert_handler = AlertWebhookHandler(webhook_url=webhook_url_str, min_level="ERROR")
        logger.add(
            alert_handler.write, # Passe la méthode `write` de l'instance du handler
            level="ERROR", # Ce niveau est une porte d'entrée, le filtrage fin se fait dans le handler
            enqueue=True,
            # Le format n'est pas directement utilisé par le handler personnalisé, mais peut être utile pour le debug
            # format="{message}", # Le handler formate lui-même le message pour le webhook
        )

    logger.info(f"Logging system configured. Console: {monitoring_config.log_level_console}, File: {monitoring_config.log_level_file if monitoring_config.log_to_file_enabled else 'Disabled'}.")

# Exemple d'utilisation (à appeler au démarrage de l'application)
# if __name__ == "__main__":
#     # Ceci est un exemple. Normalement, `settings` serait importé de `src.core.config`
#     # et `setup_logging` appelé dans le point d'entrée principal de l'application.
#     from src.core.config import settings as app_settings # Assurez-vous que config.py est accessible
    
#     # Pour cet exemple, nous allons créer un mock Settings si config.py n'est pas exécutable directement
#     # ou si l'import circulaire pose problème dans ce contexte d'exemple.
#     class MockMonitoringConfig:
#         log_level_console = "DEBUG"
#         log_level_file = "DEBUG"
#         log_to_file_enabled = True
#         log_rotation = "10 MB"
#         log_retention = "3 days"
#         log_compression = "zip"
#         alert_webhook_url = None # ou "https://hooks.slack.com/services/..." pour tester

#     class MockSettings:
#         monitoring = MockMonitoringConfig()

#     mock_settings = MockSettings()
#     setup_logging(mock_settings)

#     logger.debug("Ceci est un message de debug.")
#     logger.info("Ceci est un message d'info.")
#     logger.success("Opération réussie !") # Loguru a un niveau SUCCESS
#     logger.warning("Ceci est un avertissement.")
#     logger.error("Ceci est une erreur.")
#     logger.critical("Ceci est une erreur critique.")

#     try:
#         1 / 0
#     except ZeroDivisionError:
#         logger.exception("Une exception ZeroDivisionError s'est produite !")
    
#     logger.bind(user_id=123, transaction_id="XYZ789").info("Traitement d'une transaction utilisateur.")

#     if mock_settings.monitoring.alert_webhook_url:
#         logger.error("Test d'alerte webhook pour une erreur.")

