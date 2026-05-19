import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

def get_crawler_session():
    session = requests.Session()
    
    # 1. تنظیمات پروکسی
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url:
        session.proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        print(f"🌐 Network: Using Proxy ({proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url})")
    else:
        print("🌐 Network: Using Direct Connection (No Proxy)")

    # 2. تنظیمات تلاش مجدد (Retry)
    retries = Retry(
        total=3, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 3. هدرهای استاندارد برای جلوگیری از بلاک شدن
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    })
    
    return session
