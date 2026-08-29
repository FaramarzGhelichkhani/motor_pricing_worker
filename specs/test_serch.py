import json
import argparse
import time
import os
import re
from google import genai
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from google.genai import types

# ========================================================
# ۱. ایمپورت‌های مربوط به محیط کراولر شما
# ========================================================
from core.models import Motorcycle, CrawlStatus 
from core.database import SessionLocal 
from core.config import DJANGO_DB_URL, GEMINI_API_KEY_RESERVE, MODEL_NAME

# ========================================================
# ۲. تنظیمات دیتابیس (جنگو) و جمنای
# ========================================================
django_engine = create_engine(DJANGO_DB_URL)
client = genai.Client(api_key=GEMINI_API_KEY_RESERVE)

# فقط این کلیدها مجاز هستند (دقیقاً کلیدهای استفاده شده در قالب سایت شما)
ALLOWED_KEYS = [
    "capacity", "max_power", "max_torque", "weight", "cooling_system", 
    "fuel_capacity", "engine_type", "bore_x_stroke", "compression_ratio", 
    "fuel_system", "starter", "transmission", "clutch", "final_drive", 
    "frame", "front_suspension", "rear_suspension", "front_brakes", 
    "rear_brakes", "front_tyre", "rear_tyre", "seat_height"
]

def call_gemini_to_reparse_specs(raw_specs: dict) -> dict:
    """
    ارسال مشخصات به جمنای و دریافت JSON استاندارد شده
    """
    
    prompt = f"""You are an expert motorcycle mechanic and translator for the Iranian market.
Your task is to parse, format, and translate the given raw motorcycle specifications into a STRICT JSON format.

ALLOWED KEYS (DO NOT USE ANY OTHER KEYS):
{', '.join(ALLOWED_KEYS)}

RULES:
1. Output MUST be strictly valid JSON. Do not include markdown codeblocks (like ```json), just the JSON object.
2. For each allowed key present or inferable from the raw data, create a nested object with exactly two keys: "value" and "text".
3. "value": Extract ONLY the numeric value (int or float). If there is no specific numeric value (e.g. "Telescopic fork"), set it to null.
4. "text": Translate the remaining text (or unit) to Persian using professional Iranian motorcycle terminology. 
   - Convert units to Persian (e.g., "mm" -> "میلی‌متر", "HP" -> "اسب بخار", "kg" -> "کیلوگرم", "cc" -> "سی‌سی", "liter" -> "لیتر", "RPM" -> "دور در دقیقه").
5. Translate technical terms accurately for Iran:
   - "USD" or "Upside Down" -> "معکوس (USD)"
   - "Monoshock" -> "کمک‌فنر مرکزی (مونوشوک)"
   - "Liquid cooled" -> "آب خنک"
   - "Underbone" -> "آندربن (کاب)"
   - "Alloy" -> "آلیاژی"
   - "Trellis" -> "مشبک (Trellis)"
6. Convert any Persian numbers in the output "value" field back to standard English digits.
7. For composite values like "bore_x_stroke" (e.g., "73.4 x 59.0") or "compression_ratio" (e.g., "12.2:1"), set the "value" to null and put the ENTIRE translated string in the "text" field.
8. توجه داشته باش اعداد رو دقیق مطابق ورودی برای هر فیلد ذکر کنی

EXAMPLE INPUT:
{{"Engine": "155.0 ccm", "Power": "14.5 HP @ 8000 RPM", "Front suspension": "Telescopic fork", "Weight": "120 kg"}}

EXAMPLE OUTPUT:
{{
  "capacity": {{"value": 155.0, "text": "سی‌سی"}},
  "max_power": {{"value": 14.5, "text": "اسب بخار در ۸۰۰۰ دور در دقیقه"}},
  "front_suspension": {{"value": null, "text": "دوشاخ تلسکوپی"}},
  "weight": {{"value": 120, "text": "کیلوگرم"}}
}}

RAW DATA TO PROCESS:
{json.dumps(raw_specs, ensure_ascii=False)}
"""

    try:
        response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0, 
                        ),
                    )
        content = response.text.strip()
        
        # پاک‌سازی در صورتی که جمنای مارک‌داون فرستاد
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        return json.loads(content)
    except Exception as e:
        print(f"      [!] Gemini Error: {e}")
        return None


def run_reparse_pipeline(limit_count=50):
    crawler_session: Session = SessionLocal()

    print("Checking database for matching motorcycles...")
    
    # ----------------------------------------------------
    # ۱. واکشی موتورها با شرط‌های دقیق شما
    # ----------------------------------------------------
    base_query = crawler_session.query(Motorcycle).filter(
        Motorcycle.status.in_([CrawlStatus.SUCCESS, CrawlStatus.PARTIAL]),
        Motorcycle.is_url_verified == True,
        Motorcycle.specifications.isnot(None)
    )

    total_matching = base_query.count()
    print(f"\n[INFO] Total matching verified motorcycles (Should be around 137): {total_matching}")

    # ----------------------------------------------------
    # ۲. سیستم Resume (بازیابی از قطعی احتمالی)
    # ----------------------------------------------------
    processed_file = "reparsed_specs_ids.txt"
    processed_ids = set()
    
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            for line in f:
                if line.strip().isdigit():
                    processed_ids.add(int(line.strip()))
                    
    print(f"[INFO] Already processed in previous runs: {len(processed_ids)}")

    # فیلتر کردن موتورهایی که قبلاً پردازش شده‌اند
    if processed_ids:
        bikes_query = base_query.filter(Motorcycle.id.notin_(processed_ids))
    else:
        bikes_query = base_query

    bikes = bikes_query.limit(limit_count).all()

    if not bikes:
        print("No new motorcycles found to process. Everything is done!")
        crawler_session.close()
        return

    print(f"Fetching {len(bikes)} motorcycles in this run...\n")
    success_count = 0

    with django_engine.begin() as django_conn:
        for bike in bikes:
            brand_en = bike.brand
            model_en = bike.model_name
            
            print(f"---> Processing Specs for: {brand_en} - {model_en} (Crawler ID: {bike.id})")

            # پیدا کردن موتور در دیتابیس جنگو
            find_model_query = text("""
                SELECT m.id 
                FROM motor_motormodel m
                JOIN motor_brand b ON m.brand_id = b.id
                WHERE b.name_en = :brand_en AND m.name_en = :model_en
            """)
            model_row = django_conn.execute(find_model_query, {"brand_en": brand_en, "model_en": model_en}).fetchone()

            if not model_row:
                print("    [!] Not found in Django database. Skipping.")
                continue
            
            django_model_id = model_row[0]

            # ارسال به جمنای
            print("    Calling Gemini to restructure and translate specs...")
            gemini_output = call_gemini_to_reparse_specs(bike.specifications)

            if not gemini_output:
                print("    [!] Failed to get valid JSON from Gemini.")
                continue

            # نرمال‌سازی نهایی در پایتون (محافظت از دیتابیس)
            normalized_specs = {}
            for key in ALLOWED_KEYS:
                if key in gemini_output:
                    item = gemini_output[key]
                    if isinstance(item, dict):
                        # استخراج و تایپ کستینگ ایمن
                        val = item.get("value")
                        if val is not None:
                            try:
                                val = float(val) if '.' in str(val) else int(val)
                            except ValueError:
                                val = None
                        
                        normalized_specs[key] = {
                            "value": val,
                            "text": str(item.get("text", "")).strip()
                        }

            # آپدیت کردن فیلد specifications در جدول motor_motorspec جنگو
            if normalized_specs:
                # ----------------------------------------------------
                # تزریق نام مدل کراول شده به دیتا بدون درگیر کردن جمنای
                # ----------------------------------------------------
                normalized_specs["crawled_name"] = bike.model_name
                
                # print("    [Preview] New Spec Output:\n", json.dumps(normalized_specs, ensure_ascii=False, indent=2))

                update_spec_query = text("""
                    UPDATE motor_motorspec 
                    SET specifications = CAST(:new_specs AS jsonb)
                    WHERE motor_model_id = :model_id
                """)
                django_conn.execute(update_spec_query, {
                    "new_specs": json.dumps(normalized_specs, ensure_ascii=False),
                    "model_id": django_model_id
                })
                
                # ثبت ID در فایل برای قابلیت Resume
                with open(processed_file, "a") as f:
                    f.write(f"{bike.id}\n")
                processed_ids.add(bike.id)

                success_count += 1
                print(f"    [OK] Successfully updated specs structure for Django ID: {django_model_id}")
            else:
                print("    [!] Normalized specs were empty. Skipped.")
                
            # مکث برای جلوگیری از لیمیت شدن API جمنای
            time.sleep(3)

    crawler_session.close()
    
    print("\n================================================")
    print("Specs Restructure Process Completed!")
    print(f"Successfully reparsed and updated: {success_count} models in this run.")
    print("================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reparse raw crawler specs using Gemini to {value, text} format and update Django DB.")
    parser.add_argument("--limit", type=int, default=50, help="Number of models to process")
    args = parser.parse_args()
    run_reparse_pipeline(limit_count=args.limit)