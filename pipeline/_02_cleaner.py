# ==========================================
# Cleaners & Normalizers
# ==========================================
import re
from dictionary.mappings import GLOBAL_PLATFORM_MAP, BRAND_NAME_MAP

def normalize_digits(text):
    if not isinstance(text, str): return text
    return text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))

def extract_last_update(text, default='1405-02-23'):
    if not text: return default
    text = normalize_digits(text)
    match = re.search(r"آخرین به‌روز.*?: (.+)", text)
    if not match: return default
    
    raw = match.group(1)
    parts = raw.split("،")[0].split()
    MONTHS = {"فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4, "مرداد": 5, "شهریور": 6, 
              "مهر": 7, "آبان": 8, "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12}
    try:
        day, month, year = int(parts[0]), MONTHS.get(parts[1], 1), int(parts[2])
        return f"{year}-{month:02d}-{day:02d}"
    except: return default

def clean_year(x):
    if not x: return None
    x = normalize_digits(str(x))
    if 'قبل' in x: return 1366
    try: return int(re.search(r'\d{4}', x).group())
    except: return None

def clean_engine_volume(x):
    if not x: return None
    x = normalize_digits(str(x))
    try: return int(re.search(r'\d+', x).group())
    except: return None

def get_mileage_bucket(mileage, flag_new=0):
    if mileage is None: return 99
    if mileage == 0 and flag_new == 1: return 0
    if mileage <= 1000: return 1
    if mileage <= 5000: return 2
    if mileage <= 10000: return 3
    if mileage <= 20000: return 4
    if mileage <= 30000: return 5
    if mileage <= 50000: return 6
    if mileage <= 100000: return 7
    return 8

def validate_pre_conditions(features):
    """
    بررسی ۳ شرط حیاتی قبل از ورود به پردازش سنگین و هوش مصنوعی.
    خروجی: (is_valid, rejection_reason)
    """
    MULTI_KEYWORDS = ["اقساط", "شرایطی", "انواع", "هر موتور", "فروشگاه", "رضایت", "پیش پرداخت", "مزایده", "جشنواره"]
    TITLE_SPAM_REGEX = re.compile(r"(" + "|".join(MULTI_KEYWORDS) + r")")
    # شرط ۱: بررسی تایتل برای کلمات غیرمجاز
    title = features.get("title", "")
    if TITLE_SPAM_REGEX.search(title):
        return False, "INVALID_TITLE_SPAM"

    # شرط ۲: قیمت بین ۵۰ میلیون و ۵ میلیارد تومان
    price = features.get("price")
    if not price or price <= 50_000_000 or price >= 5_000_000_000:
        return False, "INVALID_PRICE_RANGE"

    # شرط ۳: سال تولید بزرگتر از ۱۳۹۰
    year = features.get("production_year")
    if not year or year <= 1390:
        return False, "INVALID_OLD_YEAR"

    return True, "OK"

def normalize_text(text):
    """نرمال‌سازی: اعداد به انگلیسی، حذف نیم‌فاصله و اسپیس‌های اضافی"""
    if not isinstance(text, str): return ""
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    text = text.lower().replace('\u200c', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def zeroBamaCleaner(bama_data):
    cleaned_data = []
    missing_brands = set()

    for item in bama_data:
        # --- 1. استانداردسازی برند ---
        raw_brand_en = str(item.get("brand", "")).strip().lower()
        raw_brand_fa = str(item.get("brand_fa", "")).strip().lower()

        standard_brand = BRAND_NAME_MAP.get(raw_brand_en) or BRAND_NAME_MAP.get(raw_brand_fa)
        brand_is_missing = False

        if not standard_brand:
            # اگر برند جدید بود، حرف اولش را بزرگ کن و در لیست گمشده‌ها قرار بده
            standard_brand = item.get("brand", "").capitalize()
            brand_is_missing = True
            missing_brands.add(standard_brand)

        # --- 2. ساخت رشته جستجو ---
        raw_model_en = str(item.get("model", "")).strip().lower()
        raw_model_fa = str(item.get("model_fa", "")).strip().lower()
        raw_class = str(item.get("class", "") or "").strip().lower()

        search_string = f"{raw_brand_en} {raw_brand_fa} {raw_model_en} {raw_model_fa} {raw_class}".lower()

        standard_model = None
        is_standardized = False

        # --- 3. جستجو با Regex (حل مشکل r1 و str160) ---
        for std_model_name, map_data in GLOBAL_PLATFORM_MAP.items():
            keywords = [kw.lower() for kw in map_data.get("keywords", [])]
            exclusions = [excl.lower() for excl in map_data.get("exclusions", [])]

            keyword_match = False
            # بررسی کلمات کلیدی به صورت کلمه مستقل (Word Boundary)
            for kw in keywords:
                # پترن \b کلمه را مستقل چک میکند، اما برای اطمینان از حروف فارسی از (?<![\w]) استفاده میکنیم
                pattern = r'(?<![\w])' + re.escape(kw) + r'(?![\w])'
                if re.search(pattern, search_string):
                    keyword_match = True
                    break

            if keyword_match:
                exclusion_match = False
                for excl in exclusions:
                    pattern = r'(?<![\w])' + re.escape(excl) + r'(?![\w])'
                    if re.search(pattern, search_string):
                        exclusion_match = True
                        break

                if not exclusion_match:
                    standard_model = std_model_name
                    is_standardized = True
                    break

        # --- 4. ساخت خروجی ---
        if not is_standardized:
            standard_model = item.get("model_fa") or item.get("model")

        cleaned_item = item.copy()
        cleaned_item["standard_brand"] = standard_brand
        cleaned_item["standard_model"] = standard_model
        cleaned_item["is_standardized"] = is_standardized
        cleaned_item["brand_missing"] = brand_is_missing # فلگ موقت برای استخراج برندها

        cleaned_data.append(cleaned_item)

    return cleaned_data, missing_brands

def nirooMotorCleaner(niroo_data, platform_map=GLOBAL_PLATFORM_MAP):
    cleaned_data = []

    # کلماتی که باید از تایتل حذف شوند تا به نام خالص مدل برسیم
    brands_and_extras = [
        "گلکسی", "تی وی اس", "یاماها", "دایچی", "پرواز", "احسان", "سحر", "آپاچی",
        "شکاری", "برقی", "موتور", "-", "  "
    ]

    for item in niroo_data:
        title = str(item.get("productTitle", "")) #
        slug = str(item.get("slug", "")) #[cite: 1]

        # --- 1. استخراج کلاس/ویژگی‌ها (جداسازی پرانتزهای ناقص) ---
        parts = title.split('(')
        base_title = parts[0].strip()

        extracted_classes = []
        if len(parts) > 1:
            for p in parts[1:]:
                clean_feature = p.replace(')', '').strip()
                if clean_feature:
                    extracted_classes.append(clean_feature)

        # --- 2. جستجو در مپینگ ---
        search_string = f"{title} {slug}".lower()
        standard_model = None
        is_standardized = False

        for std_model_name, map_data in platform_map.items():
            keywords = [kw.lower() for kw in map_data.get("keywords", [])]
            exclusions = [excl.lower() for excl in map_data.get("exclusions", [])]

            keyword_match = False
            for kw in keywords:
                pattern = r'(?<![\w])' + re.escape(kw) + r'(?![\w])'
                if re.search(pattern, search_string):
                    keyword_match = True
                    break

            if keyword_match:
                exclusion_match = False
                for excl in exclusions:
                    pattern = r'(?<![\w])' + re.escape(excl) + r'(?![\w])'
                    if re.search(pattern, search_string):
                        exclusion_match = True
                        break

                if not exclusion_match:
                    standard_model = std_model_name
                    is_standardized = True
                    break

        # --- 3. فرمت کردن نام مدل برای موتورهای استاندارد نشده ---
        if not is_standardized:
            raw_model = base_title
            
            for b in brands_and_extras:
                raw_model = raw_model.replace(b, " ")

            raw_model = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', raw_model)
            raw_model = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', raw_model)
            standard_model = re.sub(r'\s+', ' ', raw_model).strip().upper()

        # --- 4. ساخت خروجی نهایی ---
        cleaned_item = item.copy()
        
        # فیلدهای درخواستی
        cleaned_item["standard_brand"] = "Niroo Motor"
        cleaned_item["standard_model"] = standard_model
        cleaned_item["is_standardized"] = is_standardized
        
        # اگر کلاسی در تایتل پیدا شد، به خروجی اضافه می‌شود
        if extracted_classes:
            # می‌توانید به صورت لیست نگه دارید یا به یک استرینگ تبدیل کنید
            cleaned_item["extracted_class"] = " | ".join(extracted_classes) 

        cleaned_data.append(cleaned_item)

    return cleaned_data
