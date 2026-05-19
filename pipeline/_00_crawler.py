import json
import time
import random
import traceback
from datetime import datetime
from sqlalchemy.orm import Session
from core.models import RawListing, CrawlerState
from core.network import get_crawler_session 

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
                print("Sleeping for 2 minutes before retry...")
                time.sleep(120)

        print(f"Crawl mission (Cycle: {cycle_id}) completed successfully.")
        self._reset_state()
