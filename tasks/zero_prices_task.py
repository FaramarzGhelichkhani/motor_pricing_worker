from core.database import SessionLocal, Base, engine
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func

from core.models import MotorcycleZeroPrice
from pipeline import zeroBamaCleaner, nirooMotorCleaner
from pipeline import BamaZeroPriceCrawler, NiroomotorCrawler

import jdatetime 
from datetime import datetime
from typing import Any, List, Dict 

def init_system():
    print("🛠️ Creating PostgreSQL tables if they don't exist...")
    Base.metadata.create_all(bind=engine)


def sync_zero_prices_task():
    """
    تسک اصلی جهت فراخوانی کراولرها، تمیزسازی داده‌ها و Upsert در دیتابیس PostgreSQL
    """
    init_system()
    crawler_db: Session = SessionLocal()
    records_payload: List[Dict[str, Any]] = []
    
    today_jalali = jdatetime.date.today().strftime("%Y-%m-%d")

    try:
        print("🚀 Starting Zero Price Synchronization Task...")

        # ==========================================
        # 1. BAMA CRAWLER & CLEANER
        # ==========================================
        print("\n⏳ [1/2] Fetching Bama data...")
        bama_crawler = BamaZeroPriceCrawler()
        bama_raw_data = bama_crawler.fetch_all_prices(page_size=10)
        print(f"   => Downloaded {len(bama_raw_data)} raw records from Bama.")

        if bama_raw_data:
            bama_clean_result = zeroBamaCleaner(bama_raw_data)
            bama_cleaned_data = bama_clean_result[0] if isinstance(bama_clean_result, tuple) else bama_clean_result
            
            valid_bama_count = 0
            for item in bama_cleaned_data:
                price = item.get("price")
                if not price:
                    continue
                
                raw_title = f"{item.get('brand_fa', '')} {item.get('model_fa', '')}".strip()
                
                records_payload.append({
                    "brand_name": item.get("standard_brand", "Unknown"),
                    "model_name": item.get("standard_model", "Unknown"),
                    "motor_class": item.get("class"),
                    "price": price,
                    "jalali_date": today_jalali,
                    "source": "bama",
                    "price_provider": item.get("price_provider"),
                    "is_standardized": item.get("is_standardized", False),
                    "raw_title": raw_title,
                    "insert_date": datetime.now()
                })
                valid_bama_count += 1
                
            print(f"   => Processed and mapped {valid_bama_count} valid Bama prices.")

        # ==========================================
        # 2. NIROOMOTOR CRAWLER & CLEANER
        # ==========================================
        print("\n⏳ [2/2] Fetching Niroomotor data...")
        niroo_crawler = NiroomotorCrawler()
        niroo_raw_data = niroo_crawler.fetch_all_products(page_size=12)
        print(f"   => Downloaded {len(niroo_raw_data)} raw records from Niroomotor.")

        if niroo_raw_data:
            niroo_clean_result = nirooMotorCleaner(niroo_raw_data)
            niroo_cleaned_data = niroo_clean_result[0] if isinstance(niroo_clean_result, tuple) else niroo_clean_result
            
            valid_niroo_count = 0
            for item in niroo_cleaned_data:
                price = item.get("amount")
                if not price:
                    continue
                
                records_payload.append({
                    "brand_name": item.get("standard_brand", "Niroo Motor"),
                    "model_name": item.get("standard_model", "Unknown"),
                    "motor_class": item.get("extracted_class"),
                    "price": price,
                    "jalali_date": today_jalali,
                    "source": "niroomotor",
                    "price_provider": "قیمت نمایندگی", 
                    "is_standardized": item.get("is_standardized", False),
                    "raw_title": item.get("productTitle"),
                    "insert_date": datetime.now()
                })
                valid_niroo_count += 1
                
            print(f"   => Processed and mapped {valid_niroo_count} valid Niroomotor prices.")

        # ==========================================
        # 3. DATABASE BULK UPSERT (ON CONFLICT DO UPDATE)
        # ==========================================
        total_records = len(records_payload)
        if total_records > 0:
            print(f"\n💾 Upserting {total_records} records into PostgreSQL...")

            # یکتا‌سازی در حافظه قبل از ارسال
            unique_payload_map = {}
            for row in records_payload:
                key = (
                    row["brand_name"], 
                    row["model_name"], 
                    row["motor_class"] or "", 
                    row["jalali_date"]
                )
                unique_payload_map[key] = row 

            deduplicated_payload = list(unique_payload_map.values())

            stmt = insert(MotorcycleZeroPrice).values(deduplicated_payload)

            # انطباق دقیق با ایندکس یونیک ساخته‌شده در دیتابیس
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=[
                    MotorcycleZeroPrice.brand_name,
                    MotorcycleZeroPrice.model_name,
                    func.coalesce(MotorcycleZeroPrice.motor_class, ''),
                    MotorcycleZeroPrice.jalali_date
                ],
                set_={
                    "price": stmt.excluded.price,
                    "price_provider": stmt.excluded.price_provider,
                    "is_standardized": stmt.excluded.is_standardized,
                    "raw_title": stmt.excluded.raw_title,
                    "source": stmt.excluded.source,
                    "insert_date": stmt.excluded.insert_date,
                }
            )

            crawler_db.execute(upsert_stmt)
            crawler_db.commit()
            print("✅ Data successfully upserted into PostgreSQL (updated existing / inserted new)!")
        else:
            print("\n⚠️ No valid price records found to insert.")

    except Exception as e:
        crawler_db.rollback()
        print(f"\n❌ Task Failed! Rolled back transaction. Error details: {e}")
        
    finally:
        crawler_db.close()
        print("🏁 Task finished and database session closed.\n")

