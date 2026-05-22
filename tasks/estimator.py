import pandas as pd
import logging
import jdatetime
from ml_engine import DataPreprocessor, PriceModelTrainer
from sqlalchemy.orm import Session
from core.models import ProcessedListing 
from core.database import SessionLocal 


LOOKBACK_DAYS = 25
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
    try:
        # ==========================================
        # 1. خواندن داده‌ها از دیتابیس (Load Data)
        # ==========================================
        cutoff_date_jalali = (jdatetime.date.today() - jdatetime.timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
        
        logger.info(f"Loading raw listings from database (Jalali cutoff: {cutoff_date_jalali})...")
        
        query = db.query(ProcessedListing).filter(
            ProcessedListing.is_valid_ad == 1,
            ProcessedListing.status == 'PROCESSED_OK',
            ProcessedListing.publish_date >= cutoff_date_jalali 
        ).statement
        
        # تبدیل کوئری به Pandas DataFrame
        raw_df = pd.read_sql(query, db.bind)
        
        if raw_df.empty:
            logger.warning("No data found in the database. Pipeline stopped.")
            return {"status": "no_data"}
            
        logger.info(f"Loaded {len(raw_df)} rows from database.")

        # ==========================================
        # 2. پاک‌سازی و آماده‌سازی داده‌ها (Data Prep)
        # ==========================================
        logger.info("Starting Data Preprocessing (Imputation, Feature Engineering, Outlier Removal)...")
        preprocessor = DataPreprocessor(raw_df)
        cleaned_df = preprocessor.clean_and_prepare()
        
        if cleaned_df.empty:
            logger.warning("Dataframe is empty after preprocessing. Pipeline stopped.")
            return {"status": "no_data_after_prep"}
            
        logger.info(f"Data Preprocessing completed. {len(cleaned_df)} rows remain for training.")

        # ==========================================
        # 3. آموزش مدل و تولید سطوح قیمت (Train & Surface Gen)
        # ==========================================
        logger.info("Starting Model Training and Price Surface Generation...")
        # پاس دادن دیتافریم تمیز شده و سشن دیتابیس به ترینر
        trainer = PriceModelTrainer(df=cleaned_df, db_session=db, verbose=True)
        
        # اجرای پایپ‌لاین ترینر (آموزش + PAVA + اینزرت در دیتابیس)
        results = trainer.execute_pipeline()
        
        logger.info(f"Pipeline finished successfully. Processed Models: {results.get('processed_models', 0)}")
        return results

    except Exception as e:
        db.rollback()
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
        
    finally:
        db.close()
        logger.info("Database session closed.")
