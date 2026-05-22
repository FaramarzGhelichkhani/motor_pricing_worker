from tasks.estimator import run_price_estimation_pipeline
from core.database import engine, Base
Base.metadata.create_all(bind=engine)
if __name__ == "__main__":
    # اجرای دستی برای تست
    run_price_estimation_pipeline()
    