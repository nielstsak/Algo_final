import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy import create_engine # Nécessaire pour créer l'engine directement

from alembic import context

# --- Début des modifications pour l'intégration du projet ---

# 1. Ajouter la racine du projet au sys.path pour permettre l'import des modules du projet.
#    Le répertoire 'migrations' est généralement à la racine du projet.
#    Si 'migrations' est dans un sous-dossier, ajustez le nombre de .parent.
#    Exemple: Si projet/migrations/env.py, alors PROJECT_ROOT = Path(__file__).resolve().parent.parent
#    Si projet/alembic_config/migrations/env.py, alors PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 2. Importer la Base de vos modèles SQLAlchemy et les settings de configuration.
#    Ces imports doivent se faire APRÈS avoir ajouté PROJECT_ROOT à sys.path.
try:
    from src.core.database.models import Base as ProjectBaseModel  # Vos modèles SQLAlchemy
    from src.core.config import settings as app_settings      # Votre configuration Pydantic
    from src.core.exceptions import ConfigurationError        # Votre exception personnalisée
except ImportError as e:
    print(f"Erreur lors de l'import des modules du projet dans env.py: {e}")
    print(f"Assurez-vous que PROJECT_ROOT ({PROJECT_ROOT}) est correct et que src est un package Python.")
    sys.exit(1)


# 3. Assigner les métadonnées de vos modèles à target_metadata.
#    C'est ce qu'Alembic utilisera pour comparer avec la base de données.
target_metadata = ProjectBaseModel.metadata

# --- Fin des modifications pour l'intégration du projet ---

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line VRAIMENT setup logging according to the file named by
# the config source: fileConfig(config.config_file_name)
# Si vous utilisez Loguru et que vous l'initialisez globalement, 
# vous pourriez vouloir commenter cette ligne pour éviter les conflits.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_db_url_from_settings() -> str:
    """Récupère l'URL de la base de données depuis les settings de l'application."""
    if not app_settings.database or not app_settings.database.url:
        err_msg = "L'URL de la base de données n'est pas configurée dans les paramètres de l'application (src.core.config.settings)."
        # logger n'est pas encore configuré ici, donc print ou raise.
        print(f"ERREUR: {err_msg}") 
        raise ConfigurationError(err_msg)
    return str(app_settings.database.url) # Assurer que c'est une chaîne

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context VRAIMENT with a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_db_url_from_settings()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True, # Important pour la génération de SQL offline
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True # Recommandé pour SQLite, peut être utile pour d'autres DBs
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    db_url = get_db_url_from_settings()
    
    # Créer l'engine SQLAlchemy en utilisant l'URL des settings.
    # Les options de pooling de config.py peuvent être utilisées ici si souhaité,
    # mais pour les migrations, un engine simple est souvent suffisant.
    engine_options = {}
    if app_settings.database:
        if app_settings.database.pool_size is not None:
            engine_options['pool_size'] = app_settings.database.pool_size
        if app_settings.database.max_overflow is not None:
            engine_options['max_overflow'] = app_settings.database.max_overflow
        # Ajoutez d'autres options si nécessaire (pool_recycle, pool_pre_ping)
        # engine_options['pool_pre_ping'] = app_settings.database.pool_pre_ping_enabled

    connectable = create_engine(db_url, **engine_options)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Important pour détecter les changements de type de colonne
            render_as_batch=True # Recommandé pour SQLite et opérations complexes sur d'autres DBs
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
