from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

# ماژول‌های دیتابیس پروژه
from core.database import SessionLocal, Base, engine
from pipeline import DollarPriceCrawler

from core.models import DollarPrice

def init_system():
    print("🛠️ Checking/Creating PostgreSQL tables...")
    Base.metadata.create_all(bind=engine)

def sync_dollar_price_task():
    """
    تسک اصلی جهت دریافت آخرین قیمت دلار و Upsert آن در دیتابیس PostgreSQL
    """
    init_system()
    crawler_db: Session = SessionLocal()

    try:
        print("🚀 Starting Dollar Price Synchronization Task...")

        # ==========================================
        # 1. FETCH DATA FROM CRAWLER
        # ==========================================
        print("\n⏳ Fetching TGJU Dollar data...")
        crawler = DollarPriceCrawler()
        dollar_record = crawler.fetch_latest_price()

        if not dollar_record:
            print("⚠️ No valid dollar price data retrieved. Task aborted.")
            return

        print(f"   => Fetched Price: {dollar_record.get('price'):,} Rial")
        print(f"   => Date: {dollar_record.get('persian_date')} | Time: {dollar_record.get('time')}")

        # اضافه کردن فیلد insert_date به رکورد پیش از ارسال به دیتابیس
        dollar_record["insert_date"] = datetime.now()

        # ==========================================
        # 2. DATABASE UPSERT (ON CONFLICT DO UPDATE)
        # ==========================================
        print("\n💾 Upserting record into PostgreSQL...")
        
        # ساخت استیتمنت اینسرت پستگرس
        stmt = insert(DollarPrice).values(dollar_record)

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["persian_date"], 
            set_={
                "price": stmt.excluded.price,
                "time": stmt.excluded.time,
                "change_value": stmt.excluded.change_value,
                "change_percent": stmt.excluded.change_percent,
                "gregorian_date": stmt.excluded.gregorian_date,
                "insert_date": stmt.excluded.insert_date,
            }
        )

        crawler_db.execute(upsert_stmt)
        crawler_db.commit()
        print("✅ Dollar price successfully upserted into the database!")

    except Exception as e:
        crawler_db.rollback()
        print(f"\n❌ Task Failed! Rolled back transaction. Error details: {e}")
        
    finally:
        crawler_db.close()
        print("🏁 Task finished and database session closed.\n")
