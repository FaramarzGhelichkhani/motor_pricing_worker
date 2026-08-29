import json
import argparse
from datetime import datetime
import time
import re  # اضافه شد برای جستجوی حروف انگلیسی
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ========================================================
# ۱. ایمپورت‌های مربوط به محیط کراولر شما
# ========================================================
from core.models import Motorcycle, CrawlStatus 
from core.database import SessionLocal 
from pipeline._06_model_info_parser import MotorcycleSpecParser
from pipeline._07_model_info_gemini_reviewer import generate_motorcycle_review
from core.config import DJANGO_DB_URL

# ========================================================
# ۲. تنظیمات دیتابیس (جنگو)
# ========================================================
django_engine = create_engine(DJANGO_DB_URL)

# ========================================================
# تابع لاگ‌گیری در فایل متنی
# ========================================================
def log_issue(issue_type, brand, model, details=""):
    with open("sync_issues_report.log", "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] [{issue_type}] {brand} - {model} {details}\n")


def run_sync_pipeline(limit_count=30):
    crawler_session: Session = SessionLocal()
    parser = MotorcycleSpecParser()

    print(f"Fetching up to {limit_count} non-exported motorcycles from crawler database...")
    
    # واکشی موتورهای SUCCESS و PARTIAL که هنوز اکسپورت نشده‌اند بر اساس تعداد limit_count
    bikes = crawler_session.query(Motorcycle).filter(
        Motorcycle.status.in_([CrawlStatus.SUCCESS, CrawlStatus.PARTIAL]),
        Motorcycle.is_exported == False, Motorcycle.is_url_verified==True
    ).limit(limit_count).all()

    if not bikes:
        print("No new motorcycles to sync. Everything is up to date!")
        crawler_session.close()
        return

    print(f"Found {len(bikes)} new motorcycles to sync.")

    # متغیر شمارنده برای موتورهای دارای عکس
    bikes_with_images_count = 0

    for bike in bikes:
        print(f"\n---> Processing: {bike.brand} - {bike.model_name}")
        
        # بررسی عکس داشتن موتور و لاگ کردن آن
        if not bike.images:
            log_issue("NO_IMAGE", bike.brand, bike.model_name, "- این مدل هیچ عکسی در دیتابیس کراولر ندارد.")
        else:
            bikes_with_images_count += 1
            
        try:
            with django_engine.begin() as django_conn:
                
                # ----------------------------------------------------
                # ۱. پارس کردن دیتای کراول شده
                # ----------------------------------------------------
                parsed_specs = parser.parse_specs(bike.specifications)
                parsed_ratings = parser.parse_ratings(bike.ratings)

                # ----------------------------------------------------
                # لاگ‌گیری: بررسی تعداد کلیدها و فیلدهای ترجمه‌نشده
                # ----------------------------------------------------
                if len(parsed_specs) <= 5:
                    log_issue("LOW_SPECS", bike.brand, bike.model_name, f"- تعداد فیلدهای پارس شده بسیار کم است ({len(parsed_specs)} کلید).")

                # لیست فیلدهایی که در صفحه UI سایت نمایش داده می‌شوند
                ui_keys = [
                    "capacity", "max_power", "max_torque", "weight", "cooling_system", 
                    "fuel_capacity", "engine_type", "bore_x_stroke", "compression_ratio", 
                    "fuel_system", "starter", "transmission", "clutch", "final_drive", 
                    "frame", "front_suspension", "rear_suspension", "front_brakes", 
                    "rear_brakes", "front_tyre", "rear_tyre", "seat_height"
                ]
                
                # کلمات تخصصی انگلیسی که وجودشان در متن فارسی طبیعی و مجاز است
                allowed_eng_words = ["usd", "kyb", "ohc", "dohc", "sohc", "abs", "cbs", "efi", "m/c", "cc", "kg", "mm", "rpm", "hp", "kw"]
                untranslated_fields = []
                
                for k in ui_keys:
                    if k in parsed_specs:
                        val_str = str(parsed_specs[k])
                        val_lower = val_str.lower()
                        # پاک کردن کلمات مجاز از رشته برای تست دقیق‌تر
                        for word in allowed_eng_words:
                            val_lower = val_lower.replace(word, "")
                        
                        # اگر پس از حذف کلمات مجاز، هنوز حرف انگلیسی [a-z] در مقدار وجود داشت:
                        if re.search(r'[a-z]', val_lower):
                            untranslated_fields.append(f"{k} ({val_str})")
                            
                if untranslated_fields:
                    log_issue("UNTRANSLATED_SPEC", bike.brand, bike.model_name, f"- نیاز به ترجمه در پارسر: {', '.join(untranslated_fields)}")
                # ----------------------------------------------------

                # ----------------------------------------------------
                # ۲. تولید محتوای هوش مصنوعی (جمنای)
                # ----------------------------------------------------
                print("    Calling Gemini API...")
                ai_result = generate_motorcycle_review(
                    brand=bike.brand,
                    model_name=bike.model_name,
                    specs=parsed_specs,
                    raw_description=bike.description
                )

                if not ai_result:
                    print("    [!] AI Generation failed. Skipping this motorcycle.")
                    continue

                # ----------------------------------------------------
                # ۳. مدیریت ریتینگ‌ها و تولید overall_rating
                # ----------------------------------------------------
                if parsed_ratings and len(parsed_ratings) > 0:
                    final_ratings = parsed_ratings
                    print("    [v] Using Crawled Ratings")
                else:
                    final_ratings = ai_result.get("ratings", {})
                    print("    [v] Using AI Generated Ratings")
                
                if "overall_rating" not in final_ratings:
                    scores = [float(v) for k, v in final_ratings.items() if isinstance(v, (int, float)) and k != "overall_rating"]
                    if scores:
                        final_ratings["overall_rating"] = round(sum(scores) / len(scores), 1)
                    else:
                        final_ratings["overall_rating"] = 0.0

                # ----------------------------------------------------
                # ۴. ذخیره در دیتابیس جنگو (Upsert Brand, Model, Spec)
                # ----------------------------------------------------
                brand_en = bike.brand
                model_en = bike.model_name

                # Brand
                brand_query = text("SELECT id FROM motor_brand WHERE name_en = :name_en")
                brand_row = django_conn.execute(brand_query, {"name_en": brand_en}).fetchone()
                if brand_row:
                    brand_id = brand_row[0]
                else:
                    insert_brand = text("INSERT INTO motor_brand (name_fa, name_en) VALUES (:name_fa, :name_en) RETURNING id")
                    brand_id = django_conn.execute(insert_brand, {"name_fa": brand_en, "name_en": brand_en}).fetchone()[0]

                # Model
                model_query = text("SELECT id FROM motor_motormodel WHERE brand_id = :brand_id AND name_en = :name_en")
                model_row = django_conn.execute(model_query, {"brand_id": brand_id, "name_en": model_en}).fetchone()
                if model_row:
                    model_id = model_row[0]
                else:
                    insert_model = text("""
                        INSERT INTO motor_motormodel (brand_id, name_fa, name_en, is_original, isactive_seo, isactive_estimation, created_at, updated_at) 
                        VALUES (:brand_id, :name_fa, :name_en, False, False, False, :now, :now) RETURNING id
                    """)
                    now = datetime.utcnow()
                    model_id = django_conn.execute(insert_model, {"brand_id": brand_id, "name_fa": model_en, "name_en": model_en, "now": now}).fetchone()[0]

                # Spec
                spec_query = text("SELECT id FROM motor_motorspec WHERE motor_model_id = :model_id")
                spec_row = django_conn.execute(spec_query, {"model_id": model_id}).fetchone()
                final_ratings["pros"] = ai_result.get("pros", [])
                final_ratings["cons"] = ai_result.get("cons", [])

                spec_data = {
                    "model_id": model_id,
                    "description": bike.description or "",
                    "specifications": json.dumps(parsed_specs, ensure_ascii=False),
                    "ratings": json.dumps(final_ratings, ensure_ascii=False),
                    "ai_identity": ai_result.get("identity", ""),
                    "ai_technical": ai_result.get("technical", ""),
                    "ai_target": ai_result.get("target_audience", ""),
                    "url_specs": bike.source_url_specs or "",
                    "url_bikez": bike.source_url_bikez or "",
                    "last_updated": datetime.utcnow()
                }

                if spec_row:
                    update_spec = text("""
                        UPDATE motor_motorspec SET
                            description = :description, 
                            specifications = CAST(:specifications AS jsonb), 
                            ratings = CAST(:ratings AS jsonb),
                            ai_identity_text = :ai_identity, ai_technical_text = :ai_technical, ai_target_audience_text = :ai_target,
                            source_url_specs = :url_specs, source_url_bikez = :url_bikez, last_updated = :last_updated
                        WHERE motor_model_id = :model_id
                    """)
                    django_conn.execute(update_spec, spec_data)
                else:
                    insert_spec = text("""
                        INSERT INTO motor_motorspec (
                            motor_model_id, description, specifications, ratings, 
                            ai_identity_text, ai_technical_text, ai_target_audience_text,
                            source_url_specs, source_url_bikez, last_updated
                        ) VALUES (
                            :model_id, :description, CAST(:specifications AS jsonb), CAST(:ratings AS jsonb),
                            :ai_identity, :ai_technical, :ai_target,
                            :url_specs, :url_bikez, :last_updated
                        )
                    """)
                    django_conn.execute(insert_spec, spec_data)
                    
            # ----------------------------------------------------
            # ۵. موفقیت آمیز بود! کامیت نهایی به دیتابیس کراولر
            # ----------------------------------------------------
            bike.is_exported = True
            bike.exported_at = datetime.utcnow()
            crawler_session.commit()
            print("    [OK] Successfully saved to Django & marked as exported.")
            time.sleep(3.5)

        except Exception as e:
            crawler_session.rollback()
            print(f"    [!] Error processing {bike.brand} {bike.model_name}: {e}")

    crawler_session.close()
    print("\n================================================")
    print("Sync process completed! (Images were skipped)")
    print(f"Total processed in this run: {len(bikes)}")
    print(f"Models WITH images: {bikes_with_images_count}")
    print("Check 'sync_issues_report.log' for models missing images or unparsed specs.")
    print("================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync motorcycles from crawler DB to Django DB (Without Images).")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=30, 
        help="Number of motorcycles to sync in this run (default: 30)"
    )
    
    args = parser.parse_args()
    
    run_sync_pipeline(limit_count=args.limit)

    