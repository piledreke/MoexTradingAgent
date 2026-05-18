"""SQLAlchemy models."""
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Candle(Base):
    """OHLCV candle model."""
    
    __tablename__ = "candles"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)


class Analysis(Base):
    """Analysis result model."""
    
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    analysis_type = Column(String(50), nullable=False)
    result = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
