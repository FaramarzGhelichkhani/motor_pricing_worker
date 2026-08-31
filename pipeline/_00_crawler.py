import re
import json
import time
import random
import traceback
import jdatetime
from datetime import datetime
from sqlalchemy.orm import Session
from core.models import RawListing, CrawlerState
from core.network import get_crawler_session 
from typing import Optional, Dict, Any

class DivarEnterpriseCrawler:
    def __init__(self, db: Session, target_pages=30):
        self.db = db
        self.target_pages = target_pages
        self.search_url = "https://api.divar.ir/v8/postlist/w/search"
        self.details_url = "https://api.divar.ir/v8/posts-v2/web/{token}"
        
        # استفاده از سشن مقاوم (مجهز به Retry و Proxy از فایل .env)
        self.session = get_crawler_session() 
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0"
        ]

    def _get_headers(self, referer="https://divar.ir/s/tehran/motorcycles"):
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://divar.ir",
            "Referer": referer,
            "Sec-Fetch-Site": "same-site",
            "Sec-Fetch-Mode": "cors",
            "Content-Type": "application/json"
        }

    def _get_or_create_state(self, force_new_cycle=False):
        """بازیابی وضعیت کرالر یا ساخت چرخه جدید در دیتابیس (ORM)"""
        state = self.db.query(CrawlerState).filter(CrawlerState.id == 1).first()
        
        # اگر اولین بار است، یا چرخه تمام شده، یا دستور force داده شده
        if not state or force_new_cycle or (state and state.pages_crawled >= state.target_pages):
            new_cycle_id = f"CYCLE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not state:
                state = CrawlerState(
                    id=1, current_cycle_id=new_cycle_id, pages_crawled=0, 
                    target_pages=self.target_pages, payload_json=None
                )
                self.db.add(state)
            else:
                state.current_cycle_id = new_cycle_id
                state.pages_crawled = 0
                state.target_pages = self.target_pages
                state.payload_json = None
            
            self.db.commit()
            print(f"Started NEW Cycle: {new_cycle_id}")
            return 0, None, new_cycle_id
            
        payload = json.loads(state.payload_json) if state.payload_json else None
        return state.pages_crawled, payload, state.current_cycle_id

    def _update_state(self, pages_crawled, payload_dict):
        """به‌روزرسانی پجینیشن در دیتابیس"""
        state = self.db.query(CrawlerState).filter(CrawlerState.id == 1).first()
        if state:
            state.pages_crawled = pages_crawled
            state.payload_json = json.dumps(payload_dict) if payload_dict else None
            self.db.commit()

    def _reset_state(self):
        """صفر کردن وضعیت برای چرخه بعدی"""
        state = self.db.query(CrawlerState).filter(CrawlerState.id == 1).first()
        if state:
            state.pages_crawled = 0
            state.payload_json = None
            self.db.commit()

    def _is_token_crawled(self, token):
        """بررسی وجود توکن با استفاده از SQLAlchemy"""
        exists = self.db.query(RawListing.token).filter(RawListing.token == token).first()
        return exists is not None

    def fetch_raw_detail(self, token):
        url = self.details_url.format(token=token)
        try:
            resp = self.session.get(url, headers=self._get_headers(f"https://divar.ir/v/{token}"), timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                print("Rate limit hit! Sleeping for 60 seconds...")
                time.sleep(60)
            return None
        except Exception as e:
            print(f"Error fetching detail for {token}: {str(e)}")
            return None

    def save_raw_data(self, token, raw_json_dict, cycle_id):
        """ذخیره دیتای خام با ORM"""
        if not raw_json_dict: return
            
        new_listing = RawListing(
            token=token,
            raw_json=json.dumps(raw_json_dict, ensure_ascii=False),
            is_processed=0,
        )
        self.db.merge(new_listing) 
        self.db.commit()

    def run(self, force_new_cycle=False):
        pages_crawled, saved_payload, cycle_id = self._get_or_create_state(force_new_cycle)
        print(f"🚀 Crawler Active | Cycle: {cycle_id} | Progress: {pages_crawled}/{self.target_pages} pages.")
        while pages_crawled < self.target_pages:
            request_payload = {
                "city_ids": ["1"],
                "search_data": {
                    "form_data": {
                        "data": {
                            "category": {"str": {"value": "motorcycles"}}
                        }
                    }
                }
            }
            
            if saved_payload:
                request_payload["pagination_data"] = saved_payload

            try:
                resp = self.session.post(self.search_url, json=request_payload, headers=self._get_headers(), timeout=15)
                
                if resp.status_code != 200:
                    print(f"Failed to fetch list. HTTP {resp.status_code}. Retrying in 30s...")
                    time.sleep(30)
                    continue

                data = resp.json()
                widgets = data.get("list_widgets", [])
                tokens = [w["data"]["token"] for w in widgets if w.get("widget_type") == "POST_ROW"]

                if not tokens:
                    print("No tokens found. Reached end of category.")
                    break

                print(f"Processing Page {pages_crawled + 1} | Found {len(tokens)} ads.")
                
                new_ads_count = 0
                for token in tokens:
                    if self._is_token_crawled(token):
                        continue
                        
                    raw_data = self.fetch_raw_detail(token)
                    if raw_data:
                        self.save_raw_data(token, raw_data, cycle_id)
                        new_ads_count += 1
                        
                    time.sleep(random.uniform(1.2, 2.8)) 
                
                print(f"Saved {new_ads_count} new raw listings.")

                pagination_data = data.get("pagination", {}).get("data")
                if not pagination_data:
                    print("No more pagination data available.")
                    break
                    
                pages_crawled += 1
                saved_payload = pagination_data
                self._update_state(pages_crawled, saved_payload)
                
                time.sleep(random.uniform(3.0, 5.0))

            except Exception as e:
                print(f"Critical error during crawl loop: {str(e)}")
                traceback.print_exc()
                
                # رفع مشکل PendingRollbackError و پاکسازی وضعیت تراکنش خراب
                try:
                    self.db.rollback()
                except Exception as rollback_err:
                    print(f"Rollback failed: {str(rollback_err)}")
                    
                print("Sleeping for 2 minutes before retry...")
                time.sleep(120)

        print(f"Crawl mission (Cycle: {cycle_id}) completed successfully.")
        self._reset_state()

class BamaZeroPriceCrawler:
    def __init__(self):
        self.base_url = "https://bama.ir/mad/api/price/hierarchy"
        # استفاده از سشن تنظیم شده (شامل پروکسی و مکانیزم Retry)
        self.session = get_crawler_session()
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
        ]

    def _get_request_headers(self):
        """تولید هدرهای رندوم و اختصاصی باما برای ترکیب با هدرهای سشن"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Referer": "https://bama.ir/motorcycle-price",
            "Origin": "https://bama.ir",
            "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7"
        }

    def fetch_all_prices(self, page_size=10, max_pages=25):
        """دریافت تمامی قیمت‌ها با استفاده از صفحه بندی و سشن اختصاصی"""
        all_motorcycles = []
        
        for page in range(0, max_pages + 1):
            # هدرهای جدید در هر ریکوئست با هدرهای دیفالت سشن ادغام (و در صورت تکرار، جایگزین) می‌شوند
            headers = self._get_request_headers()
            params = {
                "pageIndex": page,
                "pageSize": page_size
            }
            
            try:
                # استفاده از سشن به جای requests.get
                response = self.session.get(self.base_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                json_data = response.json()
                
                brands_data = json_data.get("data", [])
                
                if not brands_data:
                    break
                    
                for brand in brands_data:
                    items = brand.get("items", [])
                    all_motorcycles.extend(items)
                
                time.sleep(random.uniform(1.5, 3.5))
                
            except Exception as e:
                # در صورت شکست حتی پس از ۳ بار تلاش مجدد (Retry) در سشن
                print(f"Error fetching page {page}: {e}")
                break
                
        return all_motorcycles

    def save_and_report(self, data, filename="bama_zero_prices.json"):
        """چاپ تعداد رکوردها و ذخیره در فایل JSON"""
        count = len(data)
        print(f"Successfully extracted {count} rows of motorcycle prices.")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

class NiroomotorCrawler:
    def __init__(self):
        self.api_url = "https://services.niroomotorgroup.com/client/Product/List"
        self.session = get_crawler_session()
        
        # 🔑 اضافه شدن هدر Domain که سرور درخواست کرده بود
        self.session.headers.update({
            "Content-Type": "application/json",
            "Origin": "https://niroomotorgroup.com",
            "Referer": "https://niroomotorgroup.com/",
            "Domain": "niroomotorgroup.com" 
        })

    def fetch_all_products(self, page_size=12, max_pages=30):
        """دریافت تمامی محصولات سایت نیرو موتور"""
        all_products = []
        
        for page in range(1, max_pages + 1):
            payload = {
                "propertyOptionIds": [],
                "minAmount": None,
                "maxAmount": None,
                "page": str(page), 
                "pageSize": page_size,
                "order": 3
            }
            
            try:
                response = self.session.post(self.api_url, json=payload, timeout=15)
                response.raise_for_status()
                
                json_response = response.json()
                
                # استخراج دیتا (معمولاً در ساختارهای مشابه، کلید data حاوی آرایه است)
                items = json_response.get("data", []).get("items", []) if isinstance(json_response, dict) else []
                
                if not items:
                    break
                    
                all_products.extend(items)
                time.sleep(random.uniform(1.0, 2.5))
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                # چاپ خطای دقیق در صورت بروز مجدد مشکل
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Server Message: {e.response.text}")
                break
                
        return all_products

    def save_and_report(self, data, filename="niroomotor_products.json"):
        """چاپ تعداد رکوردها و ذخیره خروجی در فایل JSON"""
        count = len(data)
        print(f"Successfully extracted {count} rows of products from Niroomotor.")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

class DollarPriceCrawler:
    def __init__(self):
        # آدرس API جدید مربوط به جدول ریز معاملات امروز
        self.base_url = "https://api.tgju.org/v1/market/indicator/today-table-data/price_dollar_rl"
        self.session = get_crawler_session()
        
        # هدرهای تنظیم‌شده دقیقاً مشابه درخواست مرورگر
        self.session.headers.update({
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.tgju.org",
            "Referer": "https://www.tgju.org/",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1",
            "X-Requested-With": "XMLHttpRequest"
        })

    @staticmethod
    def _clean_html_value(value_str: Optional[str]) -> str:
        """حذف تمامی تگ‌های HTML و کاما برای اعداد خالص"""
        if not value_str or str(value_str).strip() == "-":
            return "0"
        clean_text = re.sub(r"<[^>]+>", "", str(value_str))
        return clean_text.replace(",", "").strip()

    @staticmethod
    def _parse_change(value_str: Optional[str]) -> float:
        """استخراج عدد تغییرات و تشخیص مثبت یا منفی بودن آن از روی کلاس HTML"""
        if not value_str or str(value_str).strip() == "-":
            return 0.0
        
        # اگر کلاس low در رشته بود یعنی قیمت افت کرده و باید منفی شود
        is_negative = 'class="low"' in str(value_str)
        
        clean_text = re.sub(r"<[^>]+>", "", str(value_str))
        clean_text = clean_text.replace(",", "").replace("%", "").strip()
        
        try:
            val = float(clean_text)
            return -val if is_negative else val
        except ValueError:
            return 0.0

    def fetch_latest_price(self) -> Optional[Dict[str, Any]]:
        """دریافت آخرین قیمت ثبت شده در امروز (بالاترین سطر)"""
        # پارامترهای ضروری برای دریافت فقط ۱ رکورد اول جدول
        params = {
            "lang": "fa",
            "draw": 1,
            "start": 0,
            "length": 1, 
            "today_table_tolerance_open": 1,
            "today_table_tolerance_yesterday": 1,
            "today_table_tolerance_range": "week"
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            raw_data = response.json()
            rows = raw_data.get("data", [])
            
            if not rows:
                print("⚠️ No data rows returned from TGJU today API.")
                return None
                
            row = rows[0]
            
            price = int(self._clean_html_value(row[0]))
            time_str = str(row[1]).strip()
            change_val = int(self._parse_change(row[2]))
            change_pct = self._parse_change(row[3])
            
            # تولید تاریخ میلادی و شمسی با فرمت YYYY-MM-DD
            gregorian_date = datetime.now().strftime("%Y-%m-%d")
            persian_date = jdatetime.date.today().strftime("%Y-%m-%d")
            
            dollar_record = {
                "price": price,
                "time": time_str,
                "change_value": change_val,
                "change_percent": change_pct,
                "gregorian_date": gregorian_date,
                "persian_date": persian_date
            }
            
            return dollar_record

        except Exception as e:
            print(f"❌ Error fetching dollar price from TGJU: {e}")
            return None

    def save_and_report(self, record: Dict[str, Any], filename: str = "dollar_latest_price.json"):
        """چاپ اطلاعات استخراج‌شده و ذخیره در فایل JSON"""
        if not record:
            print("⚠️ No record to save.")
            return

        print("========================================")
        print("💵 TGJU LATEST DOLLAR PRICE REPORT")
        print("========================================")
        print(f"Date         : {record.get('persian_date')} ({record.get('gregorian_date')})")
        print(f"Time         : {record.get('time')}")
        print(f"Price        : {record.get('price'):,} Rial")
        print(f"Change Value : {record.get('change_value'):,} Rial")
        print(f"Change Rate  : {record.get('change_percent')}%")
        print("========================================")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=4)
        print(f"📁 Successfully saved to '{filename}'")
