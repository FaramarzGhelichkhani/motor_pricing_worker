import pandas as pd
import logging
from core.config import LOOKBACK_DAYS
import jdatetime
from ml_engine._01_data_prep import DataPreprocessor
from motorcycle_surface.core import MotorcyclePriceSurface
from sqlalchemy.orm import Session
from core.models import ProcessedListing, MotorcycleZeroPrice
from core.database import DjangoSession, SessionLocal


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_price_estimation_pipeline():
    """
    اجرای کامل پایپ‌لاین هوش مصنوعی:
    1. خواندن دیتا
    2. پاک‌سازی و مهندسی فیچر
    3. آموزش مدل و ذخیره سطوح قیمتی در دیتابیس
    """
    logger.info("Starting Price Estimation Pipeline...")
    
    db: Session = SessionLocal()
    django_db = DjangoSession()
    try:
        # ==========================================
        # 1. خواندن داده‌ها از دیتابیس (Load Data)
        # ==========================================
        query_ids = """
           SELECT  m.id as motorcycle_model_id , b.name_en as real_brand, m.name_en as real_model , m.is_original as is_original, m.parent_id as parent_id
            FROM motor_motormodel m 
            JOIN motor_brand b ON m.brand_id = b.id
            where m.parent_id is not null
        """
        django_models_df = pd.read_sql(query_ids, django_db.bind)


        cutoff_date_jalali = (jdatetime.date.today() - jdatetime.timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        
        logger.info(f"Loading raw listings from database (Jalali cutoff: {cutoff_date_jalali})...")
        
        query = db.query(ProcessedListing).filter(
            ProcessedListing.is_valid_ad == 1,
            ProcessedListing.status == 'PROCESSED_OK',
            ProcessedListing.publish_date >= cutoff_date_jalali 
        ).statement
        
        # تبدیل کوئری به Pandas DataFrame
        raw_df = pd.read_sql(query, db.bind)

        raw_df = raw_df.merge(django_models_df, on=["real_brand", "real_model"], how="left")
        # raw_df["is_original"] = raw_df["is_original"].fillna(0).astype(int)

        # zero data
        from datetime import datetime, timedelta
        threshold_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
        query = db.query(MotorcycleZeroPrice).filter(
                    MotorcycleZeroPrice.is_standardized == True,
                    MotorcycleZeroPrice.insert_date >= threshold_date 
                ).statement
                
        zero_df = pd.read_sql(query, db.bind)

        if raw_df.empty:
            logger.warning("No data found in the database. Pipeline stopped.")
            return {"status": "no_data"}
            
        logger.info(f"Loaded {len(raw_df)} rows from database.")

        # ==========================================
        # 2. پاک‌سازی و آماده‌سازی داده‌ها (Data Prep)
        # ==========================================
        logger.info("Starting Data Preprocessing (Imputation, Feature Engineering, Outlier Removal)...")
        preprocessor = DataPreprocessor(raw_df=raw_df, zero_df=zero_df)
        cleaned_df = preprocessor.enrich_feature(enable_baseline=False)
        
        if cleaned_df.empty:
            logger.warning("Dataframe is empty after preprocessing. Pipeline stopped.")
            return {"status": "no_data_after_prep"}
            
        logger.info(f"Data Preprocessing completed. {len(cleaned_df)} rows remain for training.")

        # ==========================================
        # 3. آموزش مدل و تولید سطوح قیمت (Train & Surface Gen)
        # ==========================================
        logger.info("Starting Model Training and Price Surface Generation...")
        surface_builder = MotorcyclePriceSurface(crawler_db=db, production_db=None)
        segment_metrics = surface_builder.fit(cleaned_df)
        surface = surface_builder.create_surface(segment_metrics)
        surface_builder.update_database()

        logger.info(f"Pipeline finished successfully. Processed Models: {len(surface)}")

        return True

    except Exception as e:
        db.rollback()
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
        
    finally:
        db.close()
        logger.info("Database session closed.")
