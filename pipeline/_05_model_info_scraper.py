# pipeline/_05_model_info_scraper.py
import re
import codecs
import base64
import difflib
import urllib.parse
import requests
import json
from bs4 import BeautifulSoup

class HybridMotorcycleScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        self.session.headers.update(self.headers)
        
        self.mcs_base_url = "https://www.motorcyclespecs.co.za/"
        self.bikez_base_url = "https://bikez.com"
        self.bikez_brands_url = f"{self.bikez_base_url}/brands/index.php"

    # =================================================================
    # توابع عمومی و نرمال‌سازی متن
    # =================================================================
    def _slugify(self, name: str) -> str:
        s = name.strip().lower()
        s = s.replace("&", " and ")
        s = re.sub(r"[^\w\s-]", " ", s)
        s = re.sub(r"\s+", "_", s.strip())
        return re.sub(r"_+", "_", s)

    def _clean_text(self, el) -> str:
        return re.sub(r"\s+", " ", el.get_text(" ", strip=True)) if el else ""

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r'[a-z]+|\d+', text.lower()))

    # =================================================================
    # بخش سایت MotorcycleSpecs
    # =================================================================
    def _mcs_get_main_content_area(self, soup: BeautifulSoup) -> BeautifulSoup:
        main_td = soup.find("td", id="table24")
        return main_td if main_td else soup.find("body")

    def _mcs_extract_specifications(self, content_area: BeautifulSoup) -> dict:
        specs = {}
        for table in content_area.find_all("table"):
            for row in table.find_all("tr"):
                cols = row.find_all(["td", "th"])
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True).rstrip(":")
                    value = cols[1].get_text(strip=True)
                    if key and value and len(key) < 45:
                        specs[re.sub(r"\s+", " ", key)] = re.sub(r"\s+", " ", value)
        return specs

    def _mcs_extract_images(self, content_area: BeautifulSoup, base_url: str) -> list:
        image_urls = []
        ignored = ["logo", "banner", "icon", "search", "_borders", "template", "ico"]
        for img in content_area.find_all("img", src=True):
            src = img["src"]
            if any(keyword in src.lower() for keyword in ignored):
                continue
            full_url = urllib.parse.urljoin(base_url, src)
            if full_url not in image_urls:
                image_urls.append(full_url)
        return image_urls

    def _mcs_extract_description(self, content_area: BeautifulSoup) -> str:
        temp_soup = BeautifulSoup(str(content_area), "html.parser")
        for table in temp_soup.find_all("table"):
            table.decompose()
            
        text_blocks = []
        for element in temp_soup.find_all(["p", "h1", "h2", "h3", "div"]):
            clean_text = re.sub(r"\s+", " ", element.get_text(separator=" ", strip=True))
            if clean_text and len(clean_text) > 15 and clean_text not in text_blocks:
                text_blocks.append(clean_text)
        return "\n\n".join(text_blocks)

    def scrape_motorcyclespecs(self, brand: str, model: str, direct_url: str = None) -> dict:
        matched_model_url = direct_url
        
        # اگر لینک دیفالت داده نشده بود، جستجو کن
        if not matched_model_url:
            target_full_name = f"{brand} {model}".lower().replace("-", " ").strip()
            try:
                main_resp = self.session.get(f"{self.mcs_base_url}/Manufacturer.htm", timeout=15)
            except Exception:
                return {}
                
            main_soup = BeautifulSoup(main_resp.text, "html.parser")
            brands_map = {
                a.get_text(strip=True).lower(): urllib.parse.urljoin(self.mcs_base_url, a["href"])
                for a in main_soup.find_all("a", href=True) if "bikes/" in a["href"].lower()
            }

            matched_brand = next((b for b in sorted(brands_map.keys(), key=len, reverse=True) if target_full_name.startswith(b.replace("-", " "))), None)
            if not matched_brand: return {}
                
            model_part = target_full_name.replace(matched_brand.replace("-", " "), "").strip()
            target_tokens = self._tokenize(model_part)
            matched_brand_url = brands_map[matched_brand]

            pages_to_visit = [matched_brand_url]
            visited_pages = set()
            
            while pages_to_visit:
                current_url = pages_to_visit.pop(0)
                if current_url in visited_pages: continue
                visited_pages.add(current_url)
                
                try:
                    page_resp = self.session.get(current_url, timeout=15)
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                except Exception:
                    continue
                    
                for a in page_soup.find_all("a", href=True):
                    href = a["href"]
                    if "model/" in href.lower() or "bikes/" in href.lower():
                        link_tokens = self._tokenize(a.get_text(strip=True))
                        if target_tokens.issubset(link_tokens) and len(target_tokens) > 0:
                            matched_model_url = urllib.parse.urljoin(current_url, href)
                            
                if len(visited_pages) == 1:
                    for a in page_soup.find_all("a", href=True):
                        text = a.get_text(strip=True)
                        href = a["href"]
                        if text.isdigit() and not href.startswith("http") and not href.startswith("/"):
                            next_page_url = urllib.parse.urljoin(current_url, href)
                            if next_page_url not in visited_pages and next_page_url not in pages_to_visit:
                                pages_to_visit.append(next_page_url)

        if not matched_model_url: return {}

        try:
            model_resp = self.session.get(matched_model_url, timeout=15)
        except Exception:
            return {}
            
        model_soup = BeautifulSoup(model_resp.text, "html.parser")
        content_area = self._mcs_get_main_content_area(model_soup)
        
        return {
            "source_url": matched_model_url,
            "specifications": self._mcs_extract_specifications(content_area),
            "images": self._mcs_extract_images(content_area, matched_model_url),
            "description": self._mcs_extract_description(content_area)
        }

    # =================================================================
    # بخش سایت Bikez (Rating & Fallback Specs)
    # =================================================================
    def _bikez_decode_obfuscation(self, soup: BeautifulSoup, html_text: str):
        match = re.search(r'var dataArray\s*=\s*(\[\{.*?\}\]);', html_text, re.DOTALL)
        if not match: return
        try:
            data_array = json.loads(match.group(1))
            for item in data_array:
                for cls_name, b64_val in item.items():
                    json_str = base64.b64decode(b64_val).decode('utf-8')
                    rot_str = json.loads(json_str)
                    decoded_html = codecs.encode(rot_str, 'rot_13')
                    snippet = BeautifulSoup(decoded_html, "html.parser")
                    for hidden in snippet.find_all(style=lambda v: v and 'none' in v.lower()):
                        hidden.decompose()
                    visible_text = snippet.get_text(separator=" ", strip=True)
                    visible_text = re.sub(r"\s+", " ", visible_text).strip()
                    for el in soup.find_all(class_=cls_name):
                        el.string = visible_text
        except Exception:
            pass

    def _bikez_extract_newest_exact_variant(self, series_url: str, brand: str, model: str) -> str:
        try:
            resp = self.session.get(series_url, timeout=15)
        except Exception:
            return series_url
            
        soup = BeautifulSoup(resp.text, "html.parser")
        brand_tokens = self._tokenize(brand)
        model_tokens = self._tokenize(model) - brand_tokens
        
        best_link = None
        best_score = -1000
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/motorcycles/" in href and re.search(r"_\d{4}\.php$", href):
                text = self._clean_text(a)
                anchor_tokens = self._tokenize(text) - brand_tokens
                common = model_tokens.intersection(anchor_tokens)
                
                extra_tokens = anchor_tokens - model_tokens
                score = len(common) - (1.5 * len(extra_tokens))
                
                if score > best_score:
                    best_score = score
                    best_link = urllib.parse.urljoin(self.bikez_base_url, href)
                    
        return best_link or series_url

    def scrape_bikez_data(self, brand: str, model: str, direct_url: str = None, need_specs: bool = False) -> dict:
        best_link = direct_url
        
        # اگر لینک دیفالت داده نشده بود، جستجو کن
        if not best_link:
            try:
                guess_slug = self._slugify(brand)
                guess_url = f"{self.bikez_base_url}/models/{guess_slug}_models.php"
                resp = self.session.get(guess_url, timeout=15)
                
                if resp.status_code != 200 or "model overview" not in resp.text.lower():
                    resp = self.session.get(self.bikez_brands_url, timeout=15)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    brand_links = {
                        self._clean_text(a).replace("motorcycles", "").strip().lower(): urllib.parse.urljoin(self.bikez_base_url, a["href"])
                        for a in soup.select("a[href*='/models/'][href$='_models.php']")
                    }
                    best = difflib.get_close_matches(brand.lower(), brand_links.keys(), n=1, cutoff=0.4)
                    if not best: return {}
                    brand_url = brand_links[best[0]]
                else:
                    brand_url = guess_url

                resp = self.session.get(brand_url, timeout=15)
                soup = BeautifulSoup(resp.text, "html.parser")
                
                brand_tokens = self._tokenize(brand)
                model_tokens = self._tokenize(model) - brand_tokens
                model_nums = {t for t in model_tokens if t.isdigit()}

                best_score = -1000
                is_series = False

                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "/models/" in href or "/motorcycles/" in href:
                        text = self._clean_text(a)
                        if not text: continue
                        anchor_tokens = self._tokenize(text) - brand_tokens
                        anchor_nums = {t for t in anchor_tokens if t.isdigit()}
                        
                        if model_nums and not model_nums.intersection(anchor_nums):
                            continue
                            
                        common = model_tokens.intersection(anchor_tokens)
                        extra_tokens = anchor_tokens - model_tokens
                        score = len(common) - (0.5 * len(extra_tokens))
                        
                        if score > best_score and len(common) > 0:
                            best_score = score
                            best_link = urllib.parse.urljoin(self.bikez_base_url, href)
                            is_series = ("/models/" in href and not re.search(r"_\d{4}\.php$", href))

                if not best_link: return {}

                if is_series:
                    best_link = self._bikez_extract_newest_exact_variant(best_link, brand, model)
            except Exception:
                return {}

        if not best_link:
            return {}

        # 1. استخراج مشخصات از Bikez (فقط در صورتی که نیاز باشد - Fallback)
        specs = {}
        if need_specs:
            try:
                model_resp = self.session.get(best_link, timeout=15)
                html_text = model_resp.text
                model_soup = BeautifulSoup(html_text, "html.parser")
                
                self._bikez_decode_obfuscation(model_soup, html_text)

                current_category = "General"
                invalid_keys = ["update specs", "insurance costs", "maintenance", "ask questions", "related bikes", "price as new", "sell this motorcycle"]

                for table in model_soup.find_all("table"):
                    if "InpForm" in table.get("class", []) or "footer" in table.get("class", []):
                        continue
                        
                    stop_parsing = False
                    for row in table.find_all("tr"):
                        cells = row.find_all(["td", "th"])
                        if len(cells) == 1:
                            txt = self._clean_text(cells[0])
                            if txt and len(txt) < 60:
                                current_category = txt
                                if "compare" in current_category.lower() or "further information" in current_category.lower():
                                    stop_parsing = True
                                    break
                            continue
                            
                        if len(cells) >= 2:
                            label = self._clean_text(cells[0]).rstrip(":")
                            value = self._clean_text(cells[1])
                            
                            if not label or not value: continue
                            if any(ik in label.lower() for ik in invalid_keys): continue
                            
                            if len(label) < 60:
                                cat_dict = specs.setdefault(current_category, {})
                                if "loading" not in value.lower():
                                    cat_dict[label] = value
                                elif label not in cat_dict:
                                    cat_dict[label] = value
                                    
                    if stop_parsing: break
            except Exception:
                pass

        # 2. استخراج Rating
        rating_url = best_link.replace("/motorcycles/", "/rating/")
        ratings = None
        try:
            resp = self.session.get(rating_url, timeout=15)
            full_text = self._clean_text(BeautifulSoup(resp.text, "html.parser"))
            
            if not (re.search(r"rated by\s+0\s+person", full_text, re.IGNORECASE) or "awaiting" in full_text.lower()):
                ratings = {}
                for cat in ["Overall rating", "Engine performance", "Reliability and problem-free driving", "Fun-factor", "Value for money", "Repair and maintenance costs", "Design and look", "Offroad capabilities", "Racing track capabilities", "Touring capabilities", "Low accident risk"]:
                    pattern = re.compile(
                        re.escape(cat) + r".*?([\d]+\.?\d*)\s+Average for [^\d]+?([\d]+\.?\d*)\s+Average for all[^\d]+?([\d]+\.?\d*)",
                        re.IGNORECASE | re.DOTALL,
                    )
                    m = pattern.search(full_text)
                    if m:
                        ratings[cat] = {
                            "bike": float(m.group(1)),
                            "category_avg": float(m.group(2)),
                            "overall_avg": float(m.group(3)),
                        }
                
                voters_match = re.search(r"rated by\s+(\d+)\s+person", full_text, re.IGNORECASE)
                if voters_match:
                    ratings["_voter_count"] = int(voters_match.group(1))
                    
                if not ratings:
                    ratings = None
        except Exception:
            pass

        return {
            "source_url": best_link,
            "rating_url": rating_url,
            "specifications": specs,
            "ratings": ratings
        }

    # =================================================================
    # تابع اصلی ترکیب‌کننده (Hybrid Crawler Module)
    # =================================================================
    def fetch_motorcycle_data(self, brand: str, model: str, mcs_url: str = None, bikez_url: str = None) -> dict:
        """
        دریافت اطلاعات موتور با امکان استفاده از لینک‌های دیفالت.
        اگر مشخصات در MCS یافت نشود، به عنوان Fallback از Bikez دریافت می‌شود.
        """
        print(f"\n[HYBRID FETCH] -> {brand.upper()} {model.upper()}")
        
        print(" -> Extracting Primary Specs & Images (MotorcycleSpecs)...")
        mcs_data = self.scrape_motorcyclespecs(brand, model, direct_url=mcs_url)
        
        # بررسی اینکه آیا MotorcycleSpecs مشخصات را پیدا کرد یا خیر
        has_mcs_specs = bool(mcs_data.get("specifications"))
        
        print(" -> Extracting Ratings (and Fallback Specs if needed from Bikez)...")
        # اگر mcs مشخصات نداشت، به Bikez می‌گوییم زحمت استخراج Specs را هم بکشد
        bikez_data = self.scrape_bikez_data(brand, model, direct_url=bikez_url, need_specs=not has_mcs_specs)

        # تعیین مشخصات نهایی
        final_specs = mcs_data.get("specifications", {})
        if not final_specs and bikez_data.get("specifications"):
            print(" -> [INFO] Using Bikez as fallback for specifications.")
            final_specs = bikez_data.get("specifications", {})

        result = {
            "query": {
                "brand": brand,
                "model": model
            },
            "status": "success" if (mcs_data or bikez_data) else "not_found",
            "urls": {
                "motorcyclespecs": mcs_data.get("source_url", mcs_url),
                "bikez_rating": bikez_data.get("rating_url", bikez_url)
            },
            "description": mcs_data.get("description", ""),
            "images": mcs_data.get("images", []),
            "specifications": final_specs,
            "ratings": bikez_data.get("ratings", None)
        }
        
        print(" -> Done!")
        return result
    