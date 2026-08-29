import os
import time
import requests
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload

# ========================================================
# ۱. ایمپورت‌های مربوط به دیتابیس کراولر
# (مسیر ایمپورت‌ها را بر اساس ساختار پروژه خود تنظیم کنید)
# ========================================================
from core.models import MotorcycleImage, Motorcycle
from core.database import SessionLocal

# پوشه‌ای روی لپ‌تاپ شما که فایل‌ها موقتاً آنجا دانلود می‌شوند
LOCAL_DOWNLOAD_DIR = "downloaded_images"

# پیشوند مسیر پیش‌فرض جنگو (دقیقاً همان چیزی که در upload_to دادید)
DJANGO_UPLOAD_DIR = "motor_models/gallery/"

def clean_filename(name):
    """جایگزینی اسپیس و کاراکترهای غیرمجاز برای ساخت نام فایل تمیز"""
    if not name:
        return "motor"
    name = re.sub(r'[^a-zA-Z0-9\s-]', '', name)
    return re.sub(r'[-\s]+', '-', name).strip('-').lower()

def run_image_downloader():
    # ساخت پوشه در لپ‌تاپ اگر وجود ندارد
    if not os.path.exists(LOCAL_DOWNLOAD_DIR):
        os.makedirs(LOCAL_DOWNLOAD_DIR)

    session: Session = SessionLocal()

    print("Fetching images to download...")
    
    # واکشی عکس‌هایی که لینک دارند اما هنوز دانلود نشده‌اند
    images = session.query(MotorcycleImage).options(
        joinedload(MotorcycleImage.motorcycle)
    ).filter(
        MotorcycleImage.original_url.isnot(None),
        MotorcycleImage.original_url != '',
        MotorcycleImage.is_downloaded == False
    ).all()

    if not images:
        print("No new images to download. All done!")
        session.close()
        return

    print(f"Found {len(images)} images to download.")

    # هدر مرورگر برای جلوگیری از بلاک شدن توسط سرور مبدا
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }

    success_count = 0
    error_count = 0

    for img in images:
        url = img.original_url
        brand_en = img.motorcycle.brand if img.motorcycle else "brand"
        model_en = img.motorcycle.model_name if img.motorcycle else "model"
        
        try:
            print(f"Downloading: {url[:60]}...")
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # استخراج فرمت (jpg, png و ...)
                ext = url.split('.')[-1].lower()
                ext = ext if ext in ['jpg', 'jpeg', 'png', 'webp'] else 'jpg'
                
                # ساخت نام فایل: مثلا yamaha-yz125-45.jpg
                clean_brand = clean_filename(brand_en)
                clean_model = clean_filename(model_en)
                filename = f"{clean_brand}-{clean_model}-{img.id}.{ext}"
                
                # مسیر ذخیره روی هارد لپ‌تاپ
                local_save_path = os.path.join(LOCAL_DOWNLOAD_DIR, filename)
                
                # آدرسی که باید در دیتابیس ذخیره شود (مخصوص جنگو)
                django_relative_path = f"{DJANGO_UPLOAD_DIR}{filename}"

                # ذخیره فایل در سیستم
                with open(local_save_path, 'wb') as f:
                    f.write(response.content)
                
                # آپدیت فیلدهای دیتابیس کراولر
                img.is_downloaded = True
                img.local_path = django_relative_path
                session.commit()
                
                success_count += 1
                print(f"  [+] Saved as: {filename}")
                print(f"  [+] Path set to: {django_relative_path}")
            else:
                error_count += 1
                print(f"  [!] HTTP {response.status_code} Error")
                
        except Exception as e:
            error_count += 1
            print(f"  [!] Connection Error: {e}")

        # وقفه برای بلاک نشدن آی‌پی شما
        time.sleep(1.5)

    session.close()
    print("\n================================================")
    print(f"Download Finished! Success: {success_count} | Errors: {error_count}")
    print("================================================")

if __name__ == "__main__":
    run_image_downloader()
    