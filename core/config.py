import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 30))

if ENVIRONMENT == "production":
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in production environment!")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sqlite_db_path = os.path.join(BASE_DIR, "local_dev_db.sqlite3")
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{sqlite_db_path}")

print(f"🔧 Starting in {ENVIRONMENT.upper()} mode. DB: {DATABASE_URL.split('://')[0]}")
