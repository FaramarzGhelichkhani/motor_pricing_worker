import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY_RESERVE = os.getenv("GEMINI_API_KEY_RESERVE", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 30))
TARGET_PAGE = int(os.getenv("TARGET_PAGE", 30))
RELIABLE_MAPE=int(os.getenv("RELIABLE_MAPE", 10))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 10))
IQR_K=float(os.getenv("IQR_K", 1.5))
DJANGO_DB_URL = os.getenv("DJANGO_DB_URL","")
DJANGO_DB_INTERAL_URL = os.getenv("DJANGO_DB_INTERAL_URL","")

if ENVIRONMENT == "production":
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in production environment!")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sqlite_db_path = os.path.join(BASE_DIR, "local_dev_db.sqlite3")
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{sqlite_db_path}")

print(f"🔧 Starting in {ENVIRONMENT.upper()} mode. DB: {DATABASE_URL.split('://')[0]}")
