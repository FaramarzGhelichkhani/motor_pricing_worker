import time
import schedule
from core.database import get_db
from core.config import BATCH_SIZE, TARGET_PAGE
from core.models import RawListing
from pipeline import DivarEnterpriseCrawler
from tasks.process_batch import run_processing_cycle
# from tasks.process_batch import run_processing_cycle
# from tasks.trainer import train_model  # (بعدا اضافه می‌کنید)
# from tasks.transfer import sync_to_remote_db # (بعدا اضافه می‌کنید)

def job_crawl_and_process():
    """وظیفه 1: کرال کردن و سپس پردازش آنی (ساعات 9,20، 15 و 00)"""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🕷️ Starting Crawler Job...")
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # 1. اجرای کرالر
        print("start to crawl")
        crawler = DivarEnterpriseCrawler(db=db, target_pages=TARGET_PAGE)
        crawler.run(force_new_cycle=True)
        
        print("🧠 Crawler finished. Starting AI Processor immediately...")

        total_processed = 0
        consecutive_api_errors = 0 # شمارنده خطاهای متوالی API
        
        while True:
            pending_count = db.query(RawListing).filter(RawListing.is_processed == 0).count()
            if pending_count == 0:
                print("🏁 No more raw listings left. Processing queue is completely empty!")
                break
                
            print(f"⏳ Remaining raw listings to process: {pending_count}")
            
            processed_in_batch = run_processing_cycle(db, batch_size=BATCH_SIZE)
            
            if processed_in_batch == 0:
                # اگر آگهی در دیتابیس هست اما پردازشی انجام نشده، یعنی API ارور داده است
                consecutive_api_errors += 1
                print(f"⚠️ API Error or Empty Return (Attempt {consecutive_api_errors}/5). Retrying in 15s...")
                time.sleep(15)
                
                # اگر 5 بار پشت سر هم API ارور داد، کلا بیخیال شو تا شیفت بعدی
                if consecutive_api_errors >= 5:
                    print("🚨 Too many API errors. Aborting processing shift to save system limits.")
                    break
                continue # رفتن به ابتدای حلقه (تلاش مجدد)
                
            consecutive_api_errors = 0
            total_processed += processed_in_batch
            print(f"📦 Batch done. Total processed this shift: {total_processed}")
            
            time.sleep(5) # استراحت طبیعی برای جلوگیری از Rate Limit
    except Exception as e:
        print(f"🔥 Error in Crawl/Process Job: {e}")
    finally:
        db.close()

def job_train_and_transfer():
    """وظیفه 2: آموزش مدل و انتقال به سایت (ساعت 3 بامداد)"""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🤖 Starting Training & Transfer Job...")
    
    try:
        # 1. اجرای ماژول آموزش مدل (CatBoost)
        # print("Training model...")
        # train_model()
        
        # 2. انتقال به دیتابیس سایت
        # print("Transferring data to remote site...")
        # sync_to_remote_db()
        
        print("✅ Nightly Train & Transfer Completed.")
    except Exception as e:
        print(f"🔥 Error in Train/Transfer Job: {e}")

if __name__ == "__main__":
    print("🚀 Initializing Motor Pricing Worker Daemon...")
    print(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # تنظیم برنامه‌های زمانی
    schedule.every().day.at("09:00").do(job_crawl_and_process)
    schedule.every().day.at("15:00").do(job_crawl_and_process)
    schedule.every().day.at("20:30").do(job_crawl_and_process)
    schedule.every().day.at("00:00").do(job_crawl_and_process)
    
    schedule.every().day.at("03:00").do(job_train_and_transfer)
    
    print("⏰ Scheduler is RUNNING. Waiting for next job...")
    
    # حلقه بی‌نهایت برای زنده نگه‌داشتن سرور و چک کردن زمان
    while True:
        schedule.run_pending()
        time.sleep(60) 
        