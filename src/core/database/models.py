# src/core/database/models.py
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Numeric, # For DECIMAL types
    Boolean,
    UniqueConstraint,
    Index,
    func,      # For SQL functions like NOW()
    ForeignKey,
    DateTime,   # General DateTime
    BigInteger
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql # Import du module de dialecte PostgreSQL
from sqlalchemy.sql.expression import text # For text based server defaults or conditions

# Base déclarative pour les modèles SQLAlchemy
Base = declarative_base()

class Kline1M(Base):
    """
    SQLAlchemy model for the 'klines_1m' table.
    Stores 1-minute historical kline (candlestick) data for trading pairs.
    Prices and volumes use high precision (DECIMAL(30,15)).
    """
    __tablename__ = 'klines_1m'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Unique kline identifier (primary, auto-incremented).")
    pair = Column(String(30), nullable=False, comment="Trading pair (e.g., BTCUSDC). Increased length for flexibility.")
    
    kline_open_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, comment="Kline open time (UTC, start of the interval).")
    kline_close_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, comment="Kline close time (UTC, end of the interval).")
    
    open_price = Column(Numeric(30, 15), nullable=False, comment="Kline open price. Increased precision.")
    high_price = Column(Numeric(30, 15), nullable=False, comment="Highest price during the kline period.")
    low_price = Column(Numeric(30, 15), nullable=False, comment="Lowest price during the kline period.")
    close_price = Column(Numeric(30, 15), nullable=False, comment="Kline close price.")
    
    base_asset_volume = Column(Numeric(30, 15), nullable=False, comment="Volume of the base asset (e.g., BTC for BTCUSDC) traded during the kline.")
    quote_asset_volume = Column(Numeric(30, 15), nullable=False, comment="Volume of the quote asset (e.g., USDC for BTCUSDC) traded during the kline.")
    number_of_trades = Column(Integer, nullable=False, comment="Total number of trades executed during the kline period.")
    
    taker_buy_base_asset_volume = Column(Numeric(30, 15), nullable=False, comment="Volume of the base asset bought by takers (market orders).")
    taker_buy_quote_asset_volume = Column(Numeric(30, 15), nullable=False, comment="Volume of the quote asset bought by takers.")
    
    is_kline_closed = Column(Boolean, nullable=False, comment="Boolean indicating if the kline is closed (TRUE) or still in formation (FALSE).")

    db_created_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), comment="Timestamp of the record's creation in the database.")
    db_updated_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now(), comment="Timestamp of the record's last update (auto-updated by trigger).")

    __table_args__ = (
        UniqueConstraint('pair', 'kline_open_time', name='uq_klines_1m_pair_open_time'),
        Index('idx_klines_1m_pair_open_time_desc', 'pair', text('kline_open_time DESC')), 
        Index('idx_klines_1m_open_time_desc', text('kline_open_time DESC')),
        Index('idx_klines_1m_is_kline_closed', 'is_kline_closed', postgresql_where=text('is_kline_closed = FALSE')),
    )

    def __repr__(self) -> str:
        open_time_str = self.kline_open_time.strftime('%Y-%m-%d %H:%M:%S %Z') if self.kline_open_time else 'N/A'
        return (
            f"<Kline1M(id={self.id}, pair='{self.pair}', "
            f"kline_open_time='{open_time_str}', "
            f"close_price={self.close_price}, is_kline_closed={self.is_kline_closed})>"
        )

class Strategy(Base):
    __tablename__ = 'strategies'
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_code = Column(String(100), unique=True, nullable=False, index=True, comment="Unique human-readable code for the strategy (e.g., SMA_CROSS_V1).")
    name = Column(String(255), nullable=False, comment="Descriptive name (e.g., Simple Moving Average Crossover Strategy).")
    description = Column(String, nullable=True, comment="Detailed description of the strategy's logic.")
    class_name = Column(String(255), nullable=False, comment="Python class name implementing the strategy (e.g., SMACrossStrategy).")
    default_config_json = Column(postgresql.JSONB, nullable=True, comment="Default configuration/parameters for the strategy in JSON format.")
    version = Column(Integer, nullable=False, server_default=text('1'), comment="Version of the strategy definition.")
    is_active = Column(Boolean, nullable=False, server_default=text('TRUE'), comment="Whether the strategy is currently active for use.")
    db_created_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    db_updated_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    def __repr__(self) -> str:
        return f"<Strategy(id={self.id}, strategy_code='{self.strategy_code}', name='{self.name}', version={self.version})>"

class StrategyParameterSet(Base):
    __tablename__ = 'strategy_parameters'
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete="CASCADE"), nullable=False, index=True)
    parameters_set_name = Column(String(255), nullable=False, comment="Named set of parameters (e.g., BTCUSDC_1h_Optimized_Q1_2024).")
    parameters_json = Column(postgresql.JSONB, nullable=False, comment="The actual parameters for this set in JSON format.")
    source_description = Column(String, nullable=True, comment="How this parameter set was obtained (e.g., Optuna WFO run X).")
    is_favorite = Column(Boolean, nullable=False, server_default=text('FALSE'))
    db_created_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    db_updated_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint('strategy_id', 'parameters_set_name', name='uq_strategy_parameters_set_name'),)
    def __repr__(self) -> str:
        return f"<StrategyParameterSet(id={self.id}, name='{self.parameters_set_name}', strategy_id={self.strategy_id})>"

class BacktestReport(Base):
    __tablename__ = 'backtest_reports'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)
    parameters_json = Column(postgresql.JSONB, nullable=True, comment="Strategy parameters used for this backtest.")
    report_name = Column(String(255), nullable=True, comment="Optional custom name for the backtest report.")
    pair = Column(String(30), nullable=False, index=True)
    kline_interval = Column(String(10), nullable=False, comment="Kline interval used (e.g., 1m, 1h).")
    period_start_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=False)
    period_end_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=False)
    initial_capital = Column(Numeric(30, 15))
    final_capital = Column(Numeric(30, 15))
    total_return_pct = Column(Numeric(10, 4), comment="Total return in percentage.")
    metrics_summary_json = Column(postgresql.JSONB, nullable=True, comment="Summary of key performance metrics (Sharpe, Sortino, Max Drawdown, etc.).")
    backtest_engine_config_json = Column(postgresql.JSONB, nullable=True, comment="Backtest engine configuration (fees, slippage).")
    executed_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    notes = Column(String, nullable=True)
    __table_args__ = (Index('idx_backtest_reports_strategy_pair_interval', 'strategy_id', 'pair', 'kline_interval'),)
    def __repr__(self) -> str:
        return f"<BacktestReport(id={self.id}, strategy_id={self.strategy_id}, pair='{self.pair}', interval='{self.kline_interval}')>"

class OptimizationJob(Base):
    __tablename__ = 'optimization_jobs'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_name = Column(String(255), unique=True, nullable=False, index=True, comment="Unique name for the optimization job/study.")
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)
    pair = Column(String(30), nullable=False, index=True)
    kline_interval = Column(String(10), nullable=False)
    optimization_type = Column(String(50), comment="Type of optimization (e.g., WFO, Optuna_TPE).")
    config_json = Column(postgresql.JSONB, nullable=True, comment="Optimization configuration (search space, WFO settings).")
    optuna_study_name_internal = Column(String(255), nullable=True, comment="Internal Optuna study name, if applicable.")
    status = Column(String(50), nullable=False, comment="Job status (e.g., PENDING, RUNNING, COMPLETED, FAILED).")
    best_trial_id_internal = Column(String(255), nullable=True)
    best_parameters_json = Column(postgresql.JSONB, nullable=True, comment="Best parameters found by the optimization.")
    best_objective_value = Column(Numeric, nullable=True, comment="Objective function value for the best parameters.")
    job_start_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=True)
    job_end_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=True)
    db_created_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    db_updated_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    def __repr__(self) -> str:
        return f"<OptimizationJob(id={self.id}, job_name='{self.job_name}', status='{self.status}')>"

class LiveTradeLog(Base):
    __tablename__ = 'live_trades_log'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_uuid = Column(postgresql.UUID(as_uuid=True), unique=True, nullable=False, server_default=text("gen_random_uuid()"), comment="Internal unique trade identifier.")
    binance_order_id = Column(String(255), unique=True, nullable=False, index=True, comment="Order ID from Binance.")
    binance_client_order_id = Column(String(255), unique=True, nullable=True, index=True, comment="Client Order ID sent to Binance.")
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=True, index=True)
    parameters_json_snapshot = Column(postgresql.JSONB, nullable=True, comment="Snapshot of strategy parameters at the time of the trade.")
    pair = Column(String(30), nullable=False, index=True)
    side = Column(String(10), nullable=False, comment="Trade side (e.g., BUY, SELL).")
    order_type = Column(String(20), nullable=False, comment="Order type (e.g., LIMIT, MARKET).")
    order_status = Column(String(50), nullable=False, comment="Final order status (e.g., FILLED, CANCELED).")
    quantity_ordered = Column(Numeric(30, 15), nullable=False)
    quantity_filled = Column(Numeric(30, 15), nullable=False)
    price_target = Column(Numeric(30, 15), nullable=True, comment="Target price for LIMIT orders.")
    average_fill_price = Column(Numeric(30, 15), nullable=True)
    commission_amount = Column(Numeric(30, 15), nullable=True)
    commission_asset = Column(String(10), nullable=True)
    transaction_time = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, comment="Timestamp of the trade execution on Binance.")
    pnl_realized_quote = Column(Numeric(30, 15), nullable=True, comment="Realized Profit or Loss in quote asset.")
    notes = Column(String, nullable=True)
    db_created_at = Column(postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    def __repr__(self) -> str:
        return f"<LiveTradeLog(id={self.id}, order_id='{self.binance_order_id}', pair='{self.pair}', side='{self.side}')>"

