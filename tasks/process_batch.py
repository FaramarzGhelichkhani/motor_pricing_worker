import time
from sqlalchemy.orm import Session
from core.models import RawListing, ProcessedListing
from pipeline import extract_raw_cleaned_features
from pipeline import validate_pre_conditions
from pipeline import rule_engine_guess, normalize_ai_output
from pipeline import AICritic
from pipeline import get_mileage_bucket

ai_client = AICritic()

def run_processing_cycle(db: Session, batch_size: int = 30):
    raw_ads = db.query(RawListing).filter(RawListing.is_processed == 0).limit(batch_size).all()
    if not raw_ads: return 0

    batch_payloads = []
    features_map = {}

    # فاز 1: پردازش محلی (پایتون)
    for raw in raw_ads:
        try:
            features = extract_raw_cleaned_features(raw.raw_json)

            is_valid, reject_reason = validate_pre_conditions(features)
            if not is_valid:
                _save_rejected(db, raw.token, features, reject_reason)
                continue

            sys_brand, sys_model = rule_engine_guess(features["title"], features["description"], features["brand_model_raw"])
            
            features_map[raw.token] = features
            batch_payloads.append({
                "token": raw.token,
                "title": features["title"],
                "description": features["description"],
                "sys_brand": sys_brand, "sys_model": sys_model
            })
        except Exception as e:
            print(f"Error prepping {raw.token}: {e}")

    # فاز 2: هوش مصنوعی
    print("start to ckeck with Ai")
    if not batch_payloads: return 0
    ai_result = ai_client.analyze_batch(batch_payloads)
    if not ai_result: return 0

    success_count = 0
    for res in ai_result.results:
        token = res.token
        if token not in features_map: continue
        
        try:
            features = features_map[token]
            final_model = normalize_ai_output( res.corrected_model)
            
            if not res.is_valid_ad:
                features["status"] = "INVALID_AD"
            else:
                features["status"] = "QUARANTINE" if res.confidence_score < 0.85 else "PROCESSED_OK"
            
            features["is_valid_ad"] = 1 if res.is_valid_ad else 0
            features["is_system_guess_correct"]= 1 if res.is_system_guess_correct else 0 
            features["real_brand"] = res.corrected_brand
            features["real_model"] = final_model
            features["is_copy"] = 1 if res.is_copy else 0
            features["seller_type"] = res.seller_type
            features["technical_score"] = res.technical_score
            features["is_real_price"] = 1 if res.is_real_price else 0
            
            features["flag_clean"] = 1 if res.flag_clean else 0
            features["flag_accessories"] = 1 if res.flag_accessories else 0
            features["flag_new_consumables"] = 1 if res.flag_new_consumables else 0
            features["flag_first_owner"] = 1 if res.flag_first_owner else 0
            features["flag_white_doc"] = 1 if res.flag_white_doc else 0
            features["flag_full_docs"] = 1 if res.flag_full_docs else 0
            features["flag_incomplete_docs"] = 1 if res.flag_incomplete_docs else 0
            features["flag_insurance"] = 1 if res.flag_insurance else 0
            features["flag_accident"] = 1 if res.flag_accident else 0
            features["flag_engine_issue"] = 1 if res.flag_engine_issue else 0
            features["flag_installment"] = 1 if res.flag_installment else 0
            features["flag_swap"] = 1 if res.flag_swap else 0
            features["flag_urgent"] = 1 if res.flag_urgent else 0
            features["flag_service"] = 1 if res.flag_service else 0
            features["url"] = f"https://divar.ir/v/{token}"
            flag_new = 1 if res.flag_new else 0
            features["flag_new"] = flag_new
            features["mileage_bucket"]= get_mileage_bucket(features.get("mileage"), flag_new)
            
            # ساخت رکورد ORM
            processed_obj = ProcessedListing(
                token=token,
                **features            
            )
            
            db.merge(processed_obj) 
            
            # آپدیت وضعیت به پردازش شده (1)
            raw_db_obj = db.query(RawListing).filter(RawListing.token == token).first()
            raw_db_obj.is_processed = 1
            
            db.commit() 
            success_count += 1
            
        except Exception as e:
            db.rollback() # اگر ارور داد، فقط همین تراکنش لغو می‌شود
            print(f"❌ DB Save Error on token {token}: {e}")
            try:
                failed_raw = db.query(RawListing).filter(RawListing.token == token).first()
                if failed_raw:
                    failed_raw.is_processed = -1 
                    db.commit()
            except:
                db.rollback()

    return success_count

def _save_rejected(db, token, f, reason):
    obj = ProcessedListing(token=token, status=reason, is_valid_ad=0, **f)
    db.merge(obj)
    raw_obj = db.query(RawListing).filter(RawListing.token == token).first()
    raw_obj.is_processed = 1
    db.commit()
    