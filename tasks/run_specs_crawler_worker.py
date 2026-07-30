import datetime
import time
from core.models import Motorcycle, MotorcycleImage, CrawlStatus
from pipeline import HybridMotorcycleScraper
from core.database import SessionLocal 
from sqlalchemy.orm import Session

def run_crawler_batch(batch_size=100):
    """
    تسک اجرایی: خواندن N موتور PENDING، کراول کردن و آپدیت دیتابیس
    """
    print(f"\n{'='*60}")
    print(f"[WORKER] Starting crawler batch... (Limit: {batch_size})")
    print(f"{'='*60}")

    # 1. اتصال به دیتابیس
    db_session: Session = SessionLocal()
    
    # 2. استخراج موتورهای در صف (PENDING)
    pending_motos = (
        db_session.query(Motorcycle)
        .filter(Motorcycle.status == CrawlStatus.PENDING)
        .limit(batch_size)
        .all()
    )

    if not pending_motos:
        print("[INFO] No PENDING motorcycles found in the queue. Everything is up to date!")
        db_session.close()
        return

    print(f"[INFO] Found {len(pending_motos)} pending motorcycles. Initializing scraper...")
    
    # نمونه‌سازی از کراولر
    scraper = HybridMotorcycleScraper()

    # 3. حلقه پردازش
    for idx, moto in enumerate(pending_motos, start=1):
        print(f"\n[{idx}/{len(pending_motos)}] Processing: {moto.brand} {moto.model_name}")
        
        try:
            # فراخوانی متد اصلی کراولر
            result = scraper.fetch_motorcycle_data(
                brand=moto.brand, 
                model=moto.model_name,
                mcs_url=moto.default_url_mcs,     
                bikez_url=moto.default_url_bikez  
            )
            
            # استخراج داده‌ها از خروجی
            specs = result.get("specifications", {})
            ratings = result.get("ratings") or {}
            desc = result.get("description", "")
            urls = result.get("urls", {})
            images_list = result.get("images", [])

            # === تعیین وضعیت (Status) ===
            if not specs and not ratings:
                moto.status = CrawlStatus.NOT_FOUND
            elif specs and not ratings:
                # اگر فقط مشخصات پیدا شد اما در سایت Bikez ریتینگ نداشت
                moto.status = CrawlStatus.PARTIAL
            elif specs and ratings:
                moto.status = CrawlStatus.SUCCESS
            else:
                moto.status = CrawlStatus.PARTIAL

            # === آپدیت فیلدهای موتور ===
            moto.specifications = specs
            moto.ratings = ratings
            moto.description = desc
            moto.source_url_specs = urls.get("motorcyclespecs")
            moto.source_url_bikez = urls.get("bikez_rating")
            moto.last_crawled_at = datetime.datetime.utcnow()
            moto.error_log = None  # پاک کردن خطاهای احتمالی قبلی

            # === ثبت تصاویر (بدون دانلود) ===
            # ابتدا اگر عکسی از قبل برای این موتور ثبت شده (مثلا در کراول قبلی)، آنها را پاک میکنیم تا تکراری نشود
            db_session.query(MotorcycleImage).filter_by(motorcycle_id=moto.id).delete()
            
            for img_url in images_list:
                new_image = MotorcycleImage(
                    motorcycle_id=moto.id,
                    original_url=img_url,
                    is_downloaded=False  # وضعیت دانلود فالس است تا بعدا تسک عکس‌ها سراغشان بیاید
                )
                db_session.add(new_image)

            # ذخیره تغییرات این موتور در دیتابیس
            db_session.commit()
            print(f" -> Saved successfully with status: {moto.status.name} | Found {len(images_list)} images.")

        except Exception as e:
            # در صورت بروز خطای غیرمنتظره در کد، وضعیت را ERROR می‌گذاریم
            db_session.rollback()
            moto.status = CrawlStatus.ERROR
            moto.error_log = str(e)
            moto.last_crawled_at = datetime.datetime.utcnow()
            db_session.commit()
            print(f" -> [ERROR] Failed to process {moto.brand} {moto.model_name}: {e}")

        # وقفه برای جلوگیری از بن شدن توسط سایت‌های مبدا
        time.sleep(1.5)

    db_session.close()
    print(f"\n[WORKER] Batch finished successfully.")
