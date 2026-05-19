import random
import time
from core.database import engine, Base, get_db
from tasks.process_batch import run_processing_cycle
from core.config import BATCH_SIZE

def init_system():
    print("🛠️ Creating PostgreSQL tables if they don't exist...")
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_system()
    print("🚀 Worker Started. Polling for new raw data...")
    
    db_gen = get_db()
    db = next(db_gen)
    
    while True:
        try:
            processed = run_processing_cycle(db, batch_size=BATCH_SIZE)
            if processed == 0:
                print("💤 No valid records to process. Sleeping for 10s...")
                time.sleep(10)
            else:
                print(f"✅ Successfully processed {processed} records.")
            time.sleep(random.uniform(3.0, 5.0))
        except Exception as e:
            print(f"🔥 Critical Loop Error: {e}")
            db.rollback()
            time.sleep(10)
            
            