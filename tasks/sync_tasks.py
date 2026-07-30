import pandas as pd
from sqlalchemy.orm import Session
from core.models import Motorcycle, CrawlStatus 
from core.database import DjangoSession, SessionLocal, Base, engine


def init_system():
    print("🛠️ Creating PostgreSQL tables if they don't exist...")
    Base.metadata.create_all(bind=engine)

def sync_django_to_crawler():
    """
    سرویس خواندن موتورها از سایت اصلی و تزریق آن‌ها به صف ماشین کراول
    """
    print("="*60)
    print("[SYNC] Starting Data Sync: Django (Site) -> Crawler Machine")
    print("="*60)

    init_system()
    # ---------------------------------------------------------
    # 1. خواندن داده‌ها از دیتابیس سایت (طبق کد شما)
    # ---------------------------------------------------------
    
    django_db = DjangoSession()
    try:
        print("[1/3] Loading models from Django database...")
        query_ids = """
            SELECT m.id as django_id, b.name_en as brand_name, m.name_en as model_name 
            FROM motor_motormodel m 
            JOIN motor_brand b ON m.brand_id = b.id
            JOIN motor_motormodel p1 ON m.parent_id = p1.id
            WHERE p1.parent_id IS NULL and m.is_original;
        """
        django_models_df = pd.read_sql(query_ids, django_db.bind)
        print(f" -> Found {len(django_models_df)} motorcycles in Django DB.")
        
    except Exception as e:
        print(f"[ERROR] Failed to read from Django DB: {e}")
        return

    # اگر دیتابیس جنگو خالی بود، کاری نمی‌کنیم
    if django_models_df.empty:
        print("[INFO] No data to sync.")
        return

    # ---------------------------------------------------------
    # 2. آماده‌سازی دیتابیس ماشین کراول
    # ---------------------------------------------------------
    crawler_db: Session = SessionLocal()    
    try:
        print("[2/3] Fetching existing records in Crawler Machine...")
        # تمام برندها و مدل‌های موجود را می‌گیریم تا تکراری‌ها را حذف کنیم
        existing_records = crawler_db.query(Motorcycle.brand, Motorcycle.model_name).all()
        
        # برای مقایسه دقیق‌تر، اسم‌ها را کوچک (lower) کرده و در یک مجموعه (Set) ذخیره می‌کنیم
        existing_set = {f"{brand.lower().strip()}_{model.lower().strip()}" for brand, model in existing_records}
        print(f" -> Crawler currently has {len(existing_set)} queued/processed motorcycles.")

        # ---------------------------------------------------------
        # 3. فیلتر کردن موتورهای جدید و ذخیره آن‌ها (Bulk Insert)
        # ---------------------------------------------------------
        print("[3/3] Identifying new models and queuing them...")
        new_motorcycles_to_insert = []
        
        for index, row in django_models_df.iterrows():
            brand = str(row['brand_name']).strip()
            model = str(row['model_name']).strip()
            
            # کلید یکتای مقایسه
            unique_key = f"{brand.lower()}_{model.lower()}"
            
            # اگر این موتور در دیتابیس کراولر نبود (یعنی جدید است)
            if unique_key not in existing_set:
                new_motorcycle = Motorcycle(
                    brand=brand,
                    model_name=model,
                    status=CrawlStatus.PENDING  # به صورت پیش‌فرض وضعیت پندینگ است
                )
                new_motorcycles_to_insert.append(new_motorcycle)
                
                # به Set اضافه می‌کنیم تا اگر در خود دیتابیس جنگو دیتای تکراری بود، اینجا دو بار ثبت نشود
                existing_set.add(unique_key)
        
        # ثبت یک‌جا در دیتابیس (بسیار سریع‌تر از ثبت تکی)
        if new_motorcycles_to_insert:
            crawler_db.add_all(new_motorcycles_to_insert)
            crawler_db.commit()
            print(f"[SUCCESS] {len(new_motorcycles_to_insert)} NEW motorcycles added to the Crawl Queue (PENDING).")
        else:
            print("[INFO] No new motorcycles to add. The crawler queue is already up to date.")

    except Exception as e:
        crawler_db.rollback()
        print(f"[ERROR] Transaction failed during sync: {e}")
    finally:
        crawler_db.close()
        print("[SYNC] Sync Process Finished.")
