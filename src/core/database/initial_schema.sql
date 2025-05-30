-- src/core/database/initial_schema.sql
-- Schéma initial pour la base de données PostgreSQL du projet Algo Trading Bot
-- Version 2: Alignement avec les conventions de l'API Binance et meilleures pratiques.

-- Suppression des anciennes versions de la fonction et de la table si elles existent (pour idempotence lors de l'exécution manuelle)
DROP TRIGGER IF EXISTS set_timestamp_klines_1m ON klines_1m;
DROP FUNCTION IF EXISTS trigger_set_timestamp();
DROP TABLE IF EXISTS klines_1m;

-- Fonction pour mettre à jour automatiquement la colonne updated_at
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW(); -- Utilise NOW() qui est standard SQL et TIMESTAMPTZ-aware
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Table principale pour stocker les données klines (bougies)
-- La granularité de 1 minute est implicite par le nom, mais la table pourrait stocker d'autres intervalles
-- si une colonne 'interval' était ajoutée. Pour l'instant, elle est dédiée aux klines de 1 minute.
CREATE TABLE klines_1m (
    id BIGSERIAL PRIMARY KEY,                           -- Identifiant unique auto-incrémenté
    pair VARCHAR(30) NOT NULL,                          -- La paire de trading, ex: 'BTCUSDC'. Augmentation de la longueur pour flexibilité.
    
    kline_open_time TIMESTAMPTZ NOT NULL,               -- Timestamp (UTC) de l'ouverture de la kline (début de l'intervalle)
    kline_close_time TIMESTAMPTZ NOT NULL,              -- Timestamp (UTC) de la fermeture de la kline (fin de l'intervalle)
    
    open_price DECIMAL(30,15) NOT NULL,                 -- Prix d'ouverture. Précision augmentée pour s'adapter à toutes les cryptos.
    high_price DECIMAL(30,15) NOT NULL,                 -- Prix le plus haut
    low_price DECIMAL(30,15) NOT NULL,                  -- Prix le plus bas
    close_price DECIMAL(30,15) NOT NULL,                -- Prix de fermeture
    
    base_asset_volume DECIMAL(30,15) NOT NULL,          -- Volume de l'actif de base échangé (ex: BTC pour BTCUSDC)
    quote_asset_volume DECIMAL(30,15) NOT NULL,         -- Volume de l'actif de cotation échangé (ex: USDC pour BTCUSDC)
    number_of_trades INTEGER NOT NULL,                  -- Nombre de trades durant cette kline
    
    taker_buy_base_asset_volume DECIMAL(30,15) NOT NULL, -- Volume de l'actif de base acheté par les takers
    taker_buy_quote_asset_volume DECIMAL(30,15) NOT NULL,-- Volume de l'actif de cotation acheté par les takers
    
    is_kline_closed BOOLEAN NOT NULL,                   -- Indique si la kline est clôturée (données finales) ou encore en cours

    -- Timestamps pour l'audit de la ligne elle-même dans la base de données
    db_created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    db_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Contrainte d'unicité pour éviter les doublons de klines pour une même paire au même moment d'ouverture
    CONSTRAINT uq_klines_1m_pair_open_time UNIQUE (pair, kline_open_time)
);

-- Index pour améliorer les performances des requêtes sur les klines
-- Index principal pour récupérer les klines d'une paire, triées par date (les plus récentes d'abord)
CREATE INDEX idx_klines_1m_pair_open_time_desc ON klines_1m(pair ASC, kline_open_time DESC);

-- Index sur kline_open_time seul, pour des requêtes temporelles sur toutes les paires
CREATE INDEX idx_klines_1m_open_time_desc ON klines_1m(kline_open_time DESC);

-- Index sur is_kline_closed pour filtrer rapidement les klines non finalisées si nécessaire
CREATE INDEX idx_klines_1m_is_kline_closed ON klines_1m(is_kline_closed) WHERE is_kline_closed = FALSE;

-- Trigger pour mettre à jour automatiquement la colonne db_updated_at lors d'une modification de la ligne
CREATE TRIGGER set_timestamp_klines_1m
BEFORE UPDATE ON klines_1m
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();

-- Commentaires sur la table et les colonnes pour la documentation du schéma
COMMENT ON TABLE klines_1m IS 'Stocke les données historiques des klines (bougies) de 1 minute pour les paires de trading. Les prix et volumes utilisent une haute précision (DECIMAL(30,15)).';
COMMENT ON COLUMN klines_1m.id IS 'Identifiant unique de la kline (primaire, auto-incrémenté).';
COMMENT ON COLUMN klines_1m.pair IS 'Paire de trading (ex: BTCUSDC).';
COMMENT ON COLUMN klines_1m.kline_open_time IS 'Timestamp (UTC) de l''ouverture de la kline.';
COMMENT ON COLUMN klines_1m.kline_close_time IS 'Timestamp (UTC) de la fermeture de la kline.';
COMMENT ON COLUMN klines_1m.open_price IS 'Prix d''ouverture de la kline.';
COMMENT ON COLUMN klines_1m.high_price IS 'Prix le plus haut atteint durant la kline.';
COMMENT ON COLUMN klines_1m.low_price IS 'Prix le plus bas atteint durant la kline.';
COMMENT ON COLUMN klines_1m.close_price IS 'Prix de fermeture de la kline.';
COMMENT ON COLUMN klines_1m.base_asset_volume IS 'Volume de l''actif de base échangé durant la kline.';
COMMENT ON COLUMN klines_1m.quote_asset_volume IS 'Volume de l''actif de cotation échangé durant la kline.';
COMMENT ON COLUMN klines_1m.number_of_trades IS 'Nombre de trades effectués durant la kline.';
COMMENT ON COLUMN klines_1m.taker_buy_base_asset_volume IS 'Volume de l''actif de base acheté par les takers (ordres au marché).';
COMMENT ON COLUMN klines_1m.taker_buy_quote_asset_volume IS 'Volume de l''actif de cotation acheté par les takers (ordres au marché).';
COMMENT ON COLUMN klines_1m.is_kline_closed IS 'Booléen indiquant si la kline est finale (TRUE) ou toujours en cours de formation (FALSE).';
COMMENT ON COLUMN klines_1m.db_created_at IS 'Timestamp de la création de l''enregistrement dans la base de données.';
COMMENT ON COLUMN klines_1m.db_updated_at IS 'Timestamp de la dernière mise à jour de l''enregistrement dans la base de données.';

-- Fin du schéma initial pour klines_1m
