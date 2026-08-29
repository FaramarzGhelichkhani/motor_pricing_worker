import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ========================================================
# ۱. ایمپورت‌های مربوط به محیط کراولر شما
# ========================================================
from core.models import MotorcycleImage
from core.database import SessionLocal 
from core.config import DJANGO_DB_URL

# ========================================================
# ۲. تنظیمات دیتابیس (جنگو)
# ========================================================
django_engine = create_engine(DJANGO_DB_URL)

def run_image_update_pipeline():
    crawler_session: Session = SessionLocal()

    print("Fetching local paths from crawler database...")
    
    # واکشی تمام عکس‌هایی که local_path دارند
    images_with_path = crawler_session.query(MotorcycleImage).filter(
        MotorcycleImage.local_path.isnot(None),
        MotorcycleImage.local_path != ''
    ).all()

    if not images_with_path:
        print("No downloaded images found in the crawler database.")
        crawler_session.close()
        return

    print(f"Found {len(images_with_path)} images with local paths ready to be mapped.")

    updated_count = 0
    skipped_count = 0

    with django_engine.begin() as django_conn:
        # کوئری آپدیت جنگو (فقط اگر فیلد image خالی باشه آپدیت میکنه)
        update_query = text("""
            UPDATE motor_motorimage 
            SET image = :local_path 
            WHERE original_url = :url AND (image IS NULL OR image = '')
        """)ئئ

        for img in images_with_path:
            if not img.original_url:
                continue
                
            # اجرای آپدیت بر اساس لینک اورجینال
            result = django_conn.execute(update_query, {
                "local_path": img.local_path,
                "url": img.original_url
            })
            
            # چک می‌کنیم آیا ردیفی آپدیت شد یا نه
            if result.rowcount > 0:
                updated_count += result.rowcount
                print(f"    [+] Updated: {img.local_path}")
            else:
                skipped_count += 1
                # احتمالاً یا قبلاً آپدیت شده یا اصلاً توی جنگو ردیفش ساخته نشده

    crawler_session.close()
    
    print("\n================================================")
    print("Image Path Update Completed!")
    print(f"Successfully updated (linked): {updated_count} images")
    print(f"Skipped (already linked or missing): {skipped_count} images")
    print("================================================")

if __name__ == "__main__":
    run_image_update_pipeline()
