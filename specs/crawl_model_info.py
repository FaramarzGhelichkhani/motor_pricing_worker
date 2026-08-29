from tasks.run_specs_crawler_worker import run_crawler_batch

if __name__ == "__main__":
    # با اجرای این فایل، 100 موتور کراول می‌شوند.
    run_crawler_batch(batch_size=100)
