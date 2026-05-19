from pipeline import normalize_digits
from dictionary.mappings import BRAND_NAME_MAP, GLOBAL_PLATFORM_MAP


def rule_engine_guess(title, description, brand_model_raw):
    full_text = f"{title} {description} {brand_model_raw}".lower().replace('\u200c', ' ')
    full_text =  normalize_digits(full_text)
    found_platforms = []
    
    for platform_name, data in GLOBAL_PLATFORM_MAP.items():
        if any(kw in full_text for kw in data["keywords"]) and not any(exc in full_text for exc in data["exclusions"]):
            found_platforms.append(platform_name)

    found_platforms = list(set(found_platforms))
    if len(found_platforms) != 1: return None, None
        
    platform_name = found_platforms[0]
    platform_data = GLOBAL_PLATFORM_MAP[platform_name]
    real_brand = None
    
    if platform_data.get("copy_brands"):
        for fa_name, en_name in BRAND_NAME_MAP.items():
            if en_name in platform_data["copy_brands"] and (fa_name in full_text or en_name.lower() in full_text):
                real_brand = en_name
                break
        if not real_brand and ("طرح" in full_text or "کپی" in full_text):
             real_brand = "Unknown Iranian Brand"

    if not real_brand: real_brand = platform_data["parent_brand"]

    return real_brand, platform_name


def normalize_ai_output(ai_model):
    """
    فقط مدل را کانونیکال می‌کند و برند را دقیقاً همانطور که هوش مصنوعی داده حفظ می‌کند.
    مثال 1: ("Behro", "Aerox") -> ("Behro", "Aerox 155")
    مثال 2: ("Benelli", "TNT 300") -> ("Benelli", "Benelli 300")
    """
    # 1. اگر خروجی خالی بود همان را برگردان
    if not ai_model:
        return  ai_model

    search_model = str(ai_model).lower().strip()
    
    # 2. بررسی تطبیق دقیق (اگر جمی‌نای از قبل نام کانونیکال را درست داده بود)
    for platform_name in GLOBAL_PLATFORM_MAP.keys():
        if search_model == platform_name.lower():
            return platform_name # برگرداندن با حروف بزرگ/کوچک استاندارد

    # 3. بررسی کلمات کلیدی فقط برای مدل
    found_platforms = []
    for platform_name, data in GLOBAL_PLATFORM_MAP.items():
        if any(kw in search_model for kw in data["keywords"]):
            if not any(exc in search_model for exc in data["exclusions"]):
                found_platforms.append(platform_name)

    found_platforms = list(set(found_platforms))
    
    # 4. اگر دقیقاً یک پلتفرم استاندارد پیدا شد، مدل را کانونیکال کن
    if len(found_platforms) == 1:
        canonical_model = found_platforms[0]
        return canonical_model
        
    # 5. اگر نتوانست مچ کند (مثلاً موتور جدیدی بود) به هوش مصنوعی اعتماد کن و دست نزن
    return ai_model
