# tasks/run_specs_crawler_worker.py
import datetime
import time
from core.models import Motorcycle, MotorcycleImage, CrawlStatus
from pipeline import HybridMotorcycleScraper
from core.database import SessionLocal 
from sqlalchemy.orm import Session
from sqlalchemy import or_

def run_crawler_batch(batch_size=100):
    """
    تسک اجرایی: خواندن N موتور (PENDING یا PARTIAL) تایید شده، کراول کردن و آپدیت دیتابیس
    """
    print(f"\n{'='*60}")
    print(f"[WORKER] Starting crawler batch... (Limit: {batch_size})")
    print(f"{'='*60}")

    # 1. اتصال به دیتابیس
    db_session: Session = SessionLocal()
    
    # 2. استخراج موتورهای تایید شده (Verified) که وضعیت آنها PENDING یا PARTIAL است
    pending_motos = (
        db_session.query(Motorcycle)
        .filter(Motorcycle.is_url_verified == True)
        .filter(or_(
            Motorcycle.status == CrawlStatus.PENDING,
            Motorcycle.status == CrawlStatus.PARTIAL
        ))
        .limit(batch_size)
        .all()
    )
    # pending_motos = db_session.query(Motorcycle).filter(Motorcycle.id == 296).all()
    if not pending_motos:
        print("[INFO] No pending or partial motorcycles found in the queue. Everything is up to date!")
        db_session.close()
        return

    print(f"[INFO] Found {len(pending_motos)} motorcycles to process. Initializing scraper...")
    
    # نمونه‌سازی از کراولر
    scraper = HybridMotorcycleScraper()

    # 3. حلقه پردازش
    for idx, moto in enumerate(pending_motos, start=1):
        print(f"\n[{idx}/{len(pending_motos)}] Processing: {moto.brand} {moto.model_name} (Status: {moto.status.name})")
        
        try:
            # دیتای قبلی را در نظر می‌گیریم (مخصوصاً برای PARTIAL)
            existing_specs = moto.specifications or {}
            existing_ratings = moto.ratings or {}
            existing_desc = moto.description or ""
            existing_images = [img.original_url for img in moto.images] if moto.images else []

            # فراخوانی متد اصلی کراولر
            result = scraper.fetch_motorcycle_data(
                brand=moto.brand, 
                model=moto.model_name,
                mcs_url=moto.default_url_mcs,     
                bikez_url=moto.default_url_bikez  
            )
            
            # استخراج داده‌ها از خروجی
            new_specs = result.get("specifications", {})
            new_ratings = result.get("ratings") or {}
            new_desc = result.get("description", "")
            urls = result.get("urls", {})
            new_images_list = result.get("images", [])

            # === ادغام دیتا (Smart Merge برای کیس‌های PARTIAL) ===
            # اگر دیتای جدید خالی بود اما قبلاً داشتیم، دیتای قبلی حفظ شود
            final_specs = new_specs if new_specs else existing_specs
            final_ratings = new_ratings if new_ratings else existing_ratings
            final_desc = new_desc if new_desc else existing_desc
            final_images = new_images_list if new_images_list else existing_images

            # === تعیین وضعیت (Status) ===
            if not final_specs and not final_ratings:
                moto.status = CrawlStatus.NOT_FOUND
            elif final_specs and not final_ratings:
                # اگر فقط مشخصات پیدا شد اما در سایت Bikez ریتینگ نداشت
                moto.status = CrawlStatus.PARTIAL
            elif final_specs and final_ratings:
                moto.status = CrawlStatus.SUCCESS
            else:
                moto.status = CrawlStatus.PARTIAL

            # === آپدیت فیلدهای موتور ===
            moto.specifications = final_specs
            moto.ratings = final_ratings
            moto.description = final_desc
            
            # ثبت سورس لینک‌ها اگر وجود داشتند
            if urls.get("motorcyclespecs"): moto.source_url_specs = urls.get("motorcyclespecs")
            if urls.get("bikez_rating"): moto.source_url_bikez = urls.get("bikez_rating")
            
            moto.last_crawled_at = datetime.datetime.utcnow()
            moto.error_log = None  # پاک کردن خطاهای احتمالی قبلی

            # === ثبت تصاویر ===
            # اگر عکس جدیدی پیدا شد، عکس‌های قبلی را پاک کن و جدیدها را بریز
            if new_images_list:
                db_session.query(MotorcycleImage).filter_by(motorcycle_id=moto.id).delete()
                for img_url in new_images_list:
                    new_image = MotorcycleImage(
                        motorcycle_id=moto.id,
                        original_url=img_url,
                        is_downloaded=False 
                    )
                    db_session.add(new_image)

            # ذخیره تغییرات این موتور در دیتابیس
            db_session.commit()
            print(f" -> Saved successfully with status: {moto.status.name} | Found {len(final_images)} images.")

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
    