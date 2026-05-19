from core.database import engine, Base, get_db
from pipeline import DivarEnterpriseCrawler

Base.metadata.create_all(bind=engine)
db_gen = get_db()
db = next(db_gen)

# اجرای معمولی (اگر قبلا وسط کار قطع شده باشد، ادامه می‌دهد)
crawler = DivarEnterpriseCrawler(db=db, target_pages=1)
crawler.run()

# اگر می‌خواهید یک چرخه 30 صفحه‌ای را از ابتدا به زور شروع کنید:
# crawler.run(force_new_cycle=True)