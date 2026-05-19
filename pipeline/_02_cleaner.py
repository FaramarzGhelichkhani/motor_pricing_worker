# ==========================================
# Cleaners & Normalizers
# ==========================================
import re

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
