from sqlalchemy import Column, String, Integer, Text, BigInteger, Float, Boolean, Date, DateTime, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base
from datetime import datetime
import datetime as dt

class CrawlerState(Base):
    __tablename__ = "crawler_state"
    id = Column(Integer, primary_key=True, default=1)
    current_cycle_id = Column(String, nullable=False)
    pages_crawled = Column(Integer, default=0)
    target_pages = Column(Integer, default=30)
    payload_json = Column(Text, nullable=True) # ذخیره دیکشنری پجینیشن دیوار
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class RawListing(Base):
    __tablename__ = "raw_listings"
    token = Column(String, primary_key=True, index=True)
    raw_json = Column(Text, nullable=False)
    is_processed = Column(Integer, default=0)
    crawled_at = Column(DateTime(timezone=True), server_default=func.now())

class ProcessedListing(Base):
    __tablename__ = "processed_listings"
    token = Column(String, primary_key=True)
    
    # Text Fields
    title = Column(Text)
    description = Column(Text)
    publish_date = Column(String)
    
    # Raw/System Labels
    brand_model_raw = Column(String)
    city = Column(String)
    district = Column(String)
    
    # Numeric / Core Fields
    price = Column(BigInteger)
    production_year = Column(Integer)
    mileage = Column(Integer)
    engine_volume = Column(Integer)
    
    # Categorical
    color = Column(String)
    clutch_type = Column(String)
    brake_type = Column(String)
    start_type = Column(String)
    engine_condition = Column(String)
    body_condition = Column(String)
    document_status = Column(String)
    
    # Semantic Flags (AI or Python)
    flag_clean = Column(Integer, default=0)
    flag_accessories = Column(Integer, default=0)
    flag_new_consumables = Column(Integer, default=0)
    flag_first_owner = Column(Integer, default=0)
    flag_new = Column(Integer, default=0)
    flag_white_doc = Column(Integer, default=0)
    flag_full_docs = Column(Integer, default=0)
    flag_incomplete_docs = Column(Integer, default=0)
    flag_insurance = Column(Integer, default=0)
    flag_accident = Column(Integer, default=0)
    flag_engine_issue = Column(Integer, default=0)
    flag_installment = Column(Integer, default=0)
    flag_swap = Column(Integer, default=0)
    flag_urgent = Column(Integer, default=0)
    flag_service = Column(Integer, default=0)
    
    # Final Machine Learning Labels
    real_brand = Column(String)
    real_model = Column(String)
    is_copy = Column(Integer, default=0)
    seller_type = Column(String)
    technical_score = Column(Integer)
    is_real_price = Column(Integer, default=1)
    is_valid_ad = Column(Integer, default=1)
    mileage_bucket = Column(Integer)
    status = Column(String)
    
    # System & Tracking
    is_system_guess_correct = Column(Integer)
    url = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelIndexHistory(Base):
    """
    تاریخچه اسنپ‌شات قیمت و کیفیت تخمین برای هر مدل موتور.
    معادل ModelIndexHistory در جنگو.
    """
    __tablename__ = "estimator_modelindexhistory" 
    id = Column(Integer, primary_key=True, index=True)
    
    motorcycle_model_id = Column(Integer, index=True, nullable=False)
    
    date = Column(Date, default=dt.date.today, nullable=False)
    # date = Column(String(10), nullable=False)

    # -----------------------------
    # PRICE RANGE
    # -----------------------------
    price_low = Column(BigInteger, nullable=False)
    price_mid = Column(BigInteger, nullable=False)
    price_high = Column(BigInteger, nullable=False)

    # -----------------------------
    # DYNAMIC MULTIPLIERS
    # -----------------------------
    color_coefficients = Column(JSON, default=dict)

    # -----------------------------
    # MODEL QUALITY & STATS
    # -----------------------------
    mape = Column(Float, default=0.0)
    sample_count = Column(Integer, default=0)
    algorithm = Column(String(120), nullable=False)
    is_reliable = Column(Boolean, default=False, index=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # برقراری ارتباط (Relationship) با جدول Surface
    price_surfaces = relationship("PriceSurface", back_populates="snapshot", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('real_brand', 'real_model', 'date', name='unique_model_date'),
        Index('idx_modelindexhistory_date', 'date'),
        Index('idx_modelindexhistory_bm_date', 'real_brand', 'real_model', 'date'),
    )

    def __repr__(self):
        return f"<{self.real_brand} {self.real_model} - {self.date} - MAPE: {self.mape:.1f}%>"

class PriceSurface(Base):
    """
    سطوح قیمتی تولید شده توسط مدل هوش مصنوعی.
    معادل PriceSurface در جنگو.
    """
    __tablename__ = "estimator_pricesurface" 
    id = Column(Integer, primary_key=True, index=True)
    
    snapshot_id = Column(Integer, ForeignKey('estimator_modelindexhistory.id', ondelete="CASCADE"), nullable=False, index=True)
    
    motorcycle_model_id = Column(Integer, index=True, nullable=False)

    # ابعاد سطح (Dimensions)
    year = Column(Integer, index=True, nullable=False)
    mileage_bucket = Column(Integer, index=True, nullable=False)
    color = Column(String(32), index=True, nullable=False)

    # -----------------------------
    # PREDICTED DYNAMIC INTERVAL
    # -----------------------------
    price_low = Column(BigInteger, nullable=False)
    price_mid = Column(BigInteger, nullable=False)
    price_high = Column(BigInteger, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    snapshot = relationship("ModelIndexHistory", back_populates="price_surfaces")

    __table_args__ = (
        UniqueConstraint('snapshot_id', 'real_brand', 'real_model', 'year', 'mileage_bucket', 'color', name='uniq_surface_snapshot_cell'),
        Index('surface_lookup_idx', 'real_brand', 'real_model', 'year', 'mileage_bucket', 'color'),
        Index('surface_snapshot_idx', 'snapshot_id', 'real_brand', 'real_model'),
    )

    def __repr__(self):
        return f"<{self.real_brand} {self.real_model} | y={self.year} | b={self.mileage_bucket} | Mid: {self.price_mid}>"
    