import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import jdatetime
import pandas as pd
from sqlalchemy.orm import Session

from core.config import LOOKBACK_DAYS
from core.models import ProcessedListing
from core.database import DjangoSession, SessionLocal
from ml_engine._01_data_prep import DataPreprocessor
from motorcycle_surface.core import MotorcyclePriceSurface, SurfaceConfig
from motorcycle_surface.release_gate import validate_recent_oof_release


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def run_price_estimation_pipeline():
    """Daily ads-only training, market-risk publication and database update."""
    logger.info("Starting daily price estimation pipeline...")

    # One Gregorian business date for the whole pipeline; no same-day ads.
    asof = pd.Timestamp(datetime.now(ZoneInfo("Asia/Tehran")).date())
    cutoff = asof.date() - timedelta(days=LOOKBACK_DAYS)
    cutoff_jalali = jdatetime.date.fromgregorian(date=cutoff).strftime("%Y-%m-%d")
    asof_jalali = jdatetime.date.fromgregorian(date=asof.date()).strftime("%Y-%m-%d")

    db: Session = SessionLocal()
    django_db = DjangoSession()
    try:
        # 1. Load ads and Django model IDs (same queries as the original runner).
        query_ids = """
            SELECT m.id AS motorcycle_model_id, b.name_en AS real_brand,
                   m.name_en AS real_model, m.is_original AS is_original,
                   m.parent_id AS parent_id
            FROM motor_motormodel m
            JOIN motor_brand b ON m.brand_id = b.id
            WHERE m.parent_id IS NOT NULL
        """
        django_models_df = pd.read_sql(query_ids, django_db.bind)

        query = db.query(ProcessedListing).filter(
            ProcessedListing.is_valid_ad == 1,
            ProcessedListing.status == 'PROCESSED_OK',
            ProcessedListing.publish_date >= cutoff_jalali,
            ProcessedListing.publish_date < asof_jalali,
        ).statement
        raw_df = pd.read_sql(query, db.bind)
        if raw_df.empty:
            logger.warning("No eligible listings for asof=%s", asof.date())
            return {"status": "no_data"}

        raw_df = raw_df.merge(django_models_df, on=["real_brand", "real_model"], how="left")
        if "source" in raw_df.columns:
            raw_df = raw_df.loc[raw_df["source"].fillna("ads").astype(str).str.lower().eq("ads")].copy()

        # 2. Preprocess ads only. The asof-aware Preprocessor accepts asof.
        logger.info("Preprocessing %s ads | asof=%s", len(raw_df), asof.date())
        preprocessor = DataPreprocessor(raw_df=raw_df, asof=asof)
        cleaned_df = preprocessor.enrich_feature(enable_baseline=False)
        if cleaned_df.empty:
            logger.warning("No data after preprocessing")
            return {"status": "no_data_after_prep"}

        # 3. Train and generate all age-valid mileage buckets.
        logger.info("Training motorcycle price surfaces...")
        config = SurfaceConfig(age_iterations=2000, residual_iterations=3000)
        surface_builder = MotorcyclePriceSurface(
            crawler_db=db,
            production_db=django_db,  # Only approved USED prices.
            config=config,
        )
        surface_builder.fit(cleaned_df, asof=asof, do_backtest=True)
        surface = surface_builder.create_surface(asof=asof)

        # Only the market-risk policy is allowed to set is_reliable.
        if surface.empty:
            raise ValueError("Empty surface: database not updated")
        if not surface["policy_version"].eq("USED_RELEASE_FINAL_1").all():
            raise ValueError("market-risk publication policy was not applied")
        if not surface["is_reliable"].equals(surface["publication_allowed"]):
            raise ValueError("is_reliable does not match market-risk publication decision")
        if "is_factory" in surface and surface["is_factory"].fillna(False).any():
            raise ValueError("Factory price found in used-market surface")

        # Direct daily publication; no dry-run or activation flag.
        # Fail-closed BEFORE modifying either database when quality degrades.
        published = surface.loc[surface['is_reliable']]
        if published.motorcycle_model_id.nunique() < 100:
            raise ValueError('Publication safety stop: fewer than 100 models')
        validate_recent_oof_release(surface_builder)
        surface_builder.update_database()
        published = surface.loc[surface["is_reliable"]]
        models = published["motorcycle_model_id"].nunique()
        logger.info("Done | asof=%s | total=%s | published=%s | models=%s",
                    asof.date(), len(surface), len(published), models)
        return True

    except Exception as exc:
        db.rollback()
        django_db.rollback()
        logger.exception("Pipeline failed")
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()
        django_db.close()
