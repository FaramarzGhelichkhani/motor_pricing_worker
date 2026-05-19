from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.sql import func
from core.database import Base


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
    price = Column(Integer)
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
