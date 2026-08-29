import os
import json
import requests
import time
import re
from urllib.parse import urlparse
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# ========================================================
# تنظیمات اولیه
# ========================================================
from core.models import Motorcycle, CrawlStatus
from core.database import SessionLocal
from core.config import DJANGO_DB_INTERAL_URL 



SERPER_API_KEY = "09854d8cd071d617cd62a7f78616e7e6439c8ced"


def verify_result(title, url, brand, model, site):
    """دابل چک ساختاری و محتوایی لینک‌ها"""
    if not url or not title:
        return False
        
    url_lower = url.lower()
    text_to_search = f"{title} {url}".lower()
    brand_lower = brand.lower()
    model_lower = model.lower()

    # ۱. فیلتر صفحات نامربوط
    invalid_keywords = ['/brands/', '/category/', 'index.php', 'index.html', '/contact', '/about']
    if any(kw in url_lower for kw in invalid_keywords):
        return False
        
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return False

    # ۲. فیلتر ساختار سایت
    if "bikez.com" in site and not url_lower.endswith('.php'):
        return False
    if "motorcyclespecs.co.za" in site and "/model/" not in url_lower:
        return False

    # ۳. فیلتر محتوایی (حجم انجین و کلمات)
    numbers_in_model = re.findall(r'\d+', model_lower)
    if numbers_in_model:
        for num in numbers_in_model:
            if int(num) > 9: 
                if num not in text_to_search:
                    return False

    model_words = [w for w in model_lower.split() if not w.isdigit() and len(w) > 2]
    word_matched = any(w in text_to_search for w in model_words)
    
    if brand_lower in text_to_search and word_matched:
        return True

    return False

def search_link_via_serper(brand, model, site):
    """جستجوی لینک با API"""
    query_text = f"site:{site} {brand} {model}".strip()
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query_text,
        "num": 4
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'organic' in data:
                for item in data['organic']:
                    link = item.get('link')
                    title = item.get('title')
                    if verify_result(title, link, brand, model, site):
                        return link
        return None
    except Exception as e:
        print(f"      [!] Search API error: {e}")
        return None

def clean_url(url):
    """حذف فاصله‌ها و اسلش انتهای لینک برای مقایسه دقیق‌تر"""
    if not url:
        return ""
    return url.strip().rstrip('/')

def run_auto_linker():
    start_id=1
    session: Session = SessionLocal()
    motorcycles = session.query(Motorcycle).filter(
        Motorcycle.id >= start_id
    ).order_by(Motorcycle.id.asc()).all()
    
    if not motorcycles:
        print("No motorcycles found in the database!")
        session.close()
        return

    total_bikes = len(motorcycles)
    print(f"Found {total_bikes} motorcycles to process (Starting from ID: {start_id}).")

    success_count = 0
    auto_verified_count = 0
    skipped_count = 0
    not_found_count = 0
    processed_count =0

    for bike in motorcycles:
        processed_count += 1
        brand_name = bike.brand or ""
        model_name = bike.model_name or ""
        
        print(f"\n[{processed_count}/{total_bikes}] ---> Processing ID: {bike.id} | {brand_name} {model_name}")

            
        if getattr(bike, 'is_url_verified', False):
            print("    [Skipped] URL is already verified.")
            skipped_count += 1
            continue

        # ذخیره لینک‌های قدیمی قبل از آپدیت برای مقایسه
        old_mcs = clean_url(bike.source_url_specs)
        old_bikez = clean_url(bike.source_url_bikez)

        # ۲. سرچ لینک‌های جدید
        new_mcs = search_link_via_serper(brand_name, model_name, "motorcyclespecs.co.za")
        time.sleep(0.5)
        new_bikez = search_link_via_serper(brand_name, model_name, "bikez.com")
        time.sleep(0.5)

        if new_mcs: print(f"    [+] Valid MCS Link: {new_mcs}")
        if new_bikez: print(f"    [+] Valid Bikez Link: {new_bikez}")

        try:
            # ۳. بررسی وضعیت پیدا شدن لینک‌ها
            if new_mcs or new_bikez:
                bike.default_url_mcs = new_mcs
                bike.default_url_bikez = new_bikez
                
                # مقایسه لینک جدید با لینک قدیمی (اگر یکی از آن‌ها دقیقاً همان قبلی بود -> تایید خودکار)
                new_mcs_clean = clean_url(new_mcs)
                new_bikez_clean = clean_url(new_bikez)
                
                is_identical = False
                if (new_mcs_clean and new_mcs_clean == old_mcs) or (new_bikez_clean and new_bikez_clean == old_bikez):
                    is_identical = True

                if is_identical and hasattr(bike, 'is_url_verified'):
                    bike.is_url_verified = True
                    auto_verified_count += 1
                    print("    [V] Links match the previous ones! AUTO-VERIFIED.")
                    session.commit()
                    success_count += 1
                    continue
                elif hasattr(bike, 'is_url_verified'):
                    bike.is_url_verified = False 
                    print("    [!] New links found. Waiting for manual verification.")
                
                # پاکسازی دیتای کراول قبلی
                bike.description = None
                bike.specifications = None
                bike.ratings = None
                bike.source_url_specs = None
                bike.source_url_bikez = None
                bike.error_log = None
                bike.last_crawled_at = None
                
                # تنظیم وضعیت به پندینگ برای کراول مجدد
                bike.status = CrawlStatus.PENDING
                bike.is_exported = False
                bike.exported_at = None

                session.commit()
                success_count += 1
                
            else:
                # اگر هیچ لینکی پیدا نشد، به وضعیت NOT_FOUND تغییر می‌کند
                bike.status = CrawlStatus.NOT_FOUND 
                bike.default_url_mcs = None
                bike.default_url_bikez = None
                session.commit()
                not_found_count += 1
                print(f"    [-] NO links found. Status marked as NOT_FOUND.")

        except Exception as e:
            session.rollback()
            print(f"    [!] Database error at ID {bike.id}: {e}")

    session.close()
    print(f"\n================================================")
    print(f"Process finished!")
    print(f"Total Updated: {success_count} (Auto-Verified: {auto_verified_count} | Needs Manual Verify: {success_count - auto_verified_count})")
    print(f"Skipped (Already OK): {skipped_count}")
    print(f"Marked as NOT_FOUND: {not_found_count}")
    print(f"================================================")

if __name__ == "__main__":
    run_auto_linker()
