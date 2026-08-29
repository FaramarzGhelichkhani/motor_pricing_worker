-- Active: 1779790470418@@remote-asiatech.runflare.com@30234@workerdbtzf_db
import re

class MotorcycleSpecParser:
    def __init__(self):
        # مپینگ جامع برای یکسان‌سازی کلیدها از هر دو سورس Bikez و Motorcyclespecs
        self.key_map = {
            # مشترکات
            "year": ["Model year", "Year"],
            "capacity": ["Displacement", "Capacity"],
            "engine_type": ["Engine type", "Engine details", "Engine"],
            "max_torque": ["Torque", "Max Torque"],
            "cooling_system": ["Cooling system", "Cooling System"],
            "fuel_system": ["Fuel system", "Induction"],
            "starter": ["Starter", "Starting"],
            "transmission": ["Gearbox", "Transmission"],
            "final_drive": ["Transmission type", "Driveline", "Final Drive"],
            "frame": ["Frame type", "Frame"],
            "front_suspension": ["Front suspension", "Front Suspension"],
            "rear_suspension": ["Rear suspension", "Rear Suspension"],
            "front_brakes": ["Front brakes", "Front Brakes"],
            "rear_brakes": ["Rear brakes", "Rear Brakes"],
            "wheels": ["Wheels", "Wheels"],
            "front_tyre": ["Front tire", "Front Tyre"],
            "rear_tyre": ["Rear tire", "Rear Tyre"],
            "wheelbase": ["Wheelbase", "Wheelbase"],
            "max_power": ["Power output", "Max Power", "Power"], 
            "fuel_capacity": ["Fuel capacity", "Fuel Capacity", "Fuel tank"],
            
            # موارد اختصاصی یا متفاوت در فرمت
            "weight": ["Dry weight", "Wet Weight", "Weight incl. oil, gas, etc", "Weight"],
            "compression_ratio": ["Compression Ratio", "Compression"],
            "bore_x_stroke": ["Bore x Stroke", "Bore x stroke"],
            "seat_height": ["Seat Height", "Seat height"],
            "ground_clearance": ["Ground Clearance"],
            "clutch": ["Clutch"],
            "abs": ["ABS"],
            "top_speed": ["Top speed"],
            "fuel_consumption": ["Fuel consumption"],
            "emission_standard": ["Emission details", "Greenhouse gases"],
            "colors": ["Color options"]
        }

        # مپینگ کلیدهای ریتینگ برای یکسان‌سازی با جمنای
        self.rating_map = {
            "Overall rating": "overall_rating",
            "Engine performance": "engine_performance",
            "Reliability and problem-free driving": "reliability",
            "Fun-factor": "fun_factor",
            "Value for money": "value_for_money",
            "Repair and maintenance costs": "maintenance_costs",
            "Design and look": "design",
            "Offroad capabilities": "offroad_capabilities",
            "Racing track capabilities": "racing_track_capabilities",
            "Touring capabilities": "touring_capabilities",
            "Low accident risk": "low_accident_risk"
        }

    # =========================================================
    # توابع عمومی نرمال‌سازی
    # =========================================================
    def _to_persian_digits(self, text: str) -> str:
        if not text: return ""
        en_to_fa = text.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
        return str(text).translate(en_to_fa)

    def _fix_bikez_broken_text(self, text: str) -> str:
        text = str(text)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        text = text.replace(" . ", ".")
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text) 
        replacements = {
            "Sing le": "Single", "four-st roke": "four-stroke", 
            "Ai r": "Air", "a ir": "air", "Inject ion": "Injection", 
            "Auto matic": "Automatic", "CV T": "CVT",
            "ba tter y": "battery", "Scoot er": "Scooter",
            "m ph": "mph", "m m": "mm", "k m": "km",
            "i nches": "inches", "inch e s": "inches",
            "setti ng": "setting", "pound s": "pounds",
            "br ake": "brake", "fin al": "final",
            "mul ti": "multi", "Coil spr ing": "Coil spring",
            "w ith": "with", "un derbone": "underbone",
            "rigidi ty": "rigidity", "adjustab le": "adjustable",
            "ty res": "tyres",
            "Ya m aha": "Yamaha",
            "ti r es": "tires",
            "k ick": "kick",
            "sp eed": "speed",
            "li quid": "liquid",
            "S ensitive": "Sensitive",
            "adjus table": "adjustable",
            "c radle": "cradle"
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_metric_value(self, text: str, unit: str) -> str:
        match = re.search(r'([\d\.]+)\s*(?:cc|ccm|hp|kw|nm|mm|kg|litres|liters)?', str(text), re.IGNORECASE)
        if match:
            return f"{match.group(1)} {unit}"
        return str(text)

    # =========================================================
    # پارسرهای اختصاصی
    # =========================================================
    def _parse_capacity(self, text: str) -> str:
        val = self._extract_metric_value(text, "سی‌سی")
        return self._to_persian_digits(val)

    def _parse_frame(self, text: str) -> str:
            text = str(text).lower()
            translations = {
                r'\btubular\b': 'لوله‌ای',
                r'\btrellis\b': 'مشبک (Trellis)',
                r'\blattice\b': 'مشبک',
                r'\bunderbone\b': 'آندربن (کاب)',
                r'\baluminium\b': 'آلومینیومی',
                r'\baluminum\b': 'آلومینیومی',
                r'\bsteel\b': 'فولادی',
                r'\btwin spar\b': 'جفت تیرک (Twin-spar)',
                r'\bbilateral beam\b': 'جفت تیرک (Bilateral Beam)',
                r'\bcradle\b': 'گهواره‌ای (Cradle)',
                r'\bdiamond\b': 'الماسی',
                r'\bsheet metal\b': 'ورق فلزی',
                r'\bhigh rigidity\b': 'مقاومت بالا',
                r'\bbody\b': 'بدنه',
                r'\bframe\b': 'شاسی',
                r'\bwelded reinforcements\b': 'تقویت‌شده با جوش',
                # این خطوط را به دیکشنری translations اضافه کنید:
                r'\bunderbonetype\b': 'آندربن (کاب)',
                r'\bunderbone type\b': 'آندربن (کاب)',
                r'\btype\b': 'نوع',
                r'\bsemi-double\b': 'نیمه دوبل',
                r'\bsemi double\b': 'نیمه دوبل',
                r'\bdouble cradle\b': 'گهواره‌ای دوبل',
                r'\bcradle\b': 'گهواره‌ای',
            }
            for eng, per in translations.items():
                text = re.sub(eng, per, text, flags=re.IGNORECASE)
                
            text = text.replace("with", "با")
            return self._to_persian_digits(text.strip().capitalize())

    def _parse_engine_type(self, text: str) -> str:
        text = str(text).lower()
        parts = []
        if "single" in text: parts.append("تک سیلندر")
        elif "parallel twin" in text or "inline twin" in text: parts.append("دو سیلندر خطی")
        elif "v-twin" in text: parts.append("دو سیلندر V شکل")
        elif "inline 4" in text or "in-line 4" in text or "four cylinder" in text: parts.append("چهار سیلندر خطی")
        elif "twin" in text: parts.append("دو سیلندر")
        
        if "two stroke" in text or "two-stroke" in text: parts.append("دو زمانه")
        elif "four stroke" in text or "four-stroke" in text: parts.append("چهار زمانه")
        
        if "dohc" in text: parts.append("DOHC")
        elif "ohc" in text: parts.append("OHC")
        elif "uni-cam" in text: parts.append("Unicam (SOHC)")

        if parts: return "، ".join(parts)
        return str(text)

    def _parse_power_torque(self, text: str, is_power=True) -> str:
        text = str(text).replace("))", ")")
        val_match = re.search(r'([\d\.]+)\s*(?:hp|bhp|nm)', text, re.IGNORECASE)
        rpm_match = re.search(r'@\s*([\d\.]+)\s*rpm', text, re.IGNORECASE)
        unit = "اسب بخار" if is_power else "نیوتن‌متر"
        
        if val_match:
            result = f"{val_match.group(1)} {unit}"
            if rpm_match:
                result += f" در {rpm_match.group(1)} دور در دقیقه"
            return self._to_persian_digits(result)
        return self._to_persian_digits(text)

    def _parse_cooling(self, text: str) -> str:
        text = str(text).lower()
        if "liquid" in text or "water" in text: return "آب خنک"
        if "air" in text: return "هوا خنک"
        if "oil" in text: return "روغن خنک"
        return str(text)

    def _parse_fuel_system(self, text: str) -> str:
        text = str(text).lower()
        if "inject" in text or "pgm-fi" in text: return "انژکتور"
        if "carb" in text: return "کاربراتور"
        return str(text)

    def _parse_starter(self, text: str) -> str:
        text = str(text).lower()
        if "electric" in text and "kick" in text: return "استارت الکتریکی و هندل"
        if "electric" in text: return "استارت الکتریکی"
        if "kick" in text: return "هندل"
        return str(text)

    def _parse_transmission(self, text: str) -> str:
        text = str(text).lower()
        # پشتیبانی از خط تیره: 5-speed یا 6 speed
        match = re.search(r'(\d+)\s*-?\s*speed', text)
        if match: return self._to_persian_digits(f"{match.group(1)} دنده")
        if "auto" in text: return "اتوماتیک"
        if "cvt" in text: return "گیربکس CVT"
        return self._to_persian_digits(text)

    def _parse_final_drive(self, text: str) -> str:
        text = str(text).lower()
        if "chain" in text: return "زنجیر"
        if "belt" in text: return "تسمه"
        if "shaft" in text: return "گاردان"
        return str(text)

    def _parse_clutch(self, text: str) -> str:
        text = str(text).lower()
        parts = []
        if "wet" in text: parts.append("خیس (در روغن)")
        elif "dry" in text: parts.append("خشک")
        
        if "multi" in text: parts.append("چند صفحه‌ای")
        if "slipper" in text: parts.append("اسلیپر کلاچ (لغزشی)")
        
        if parts: return "، ".join(parts)
        return str(text)

    def _parse_brakes(self, text: str) -> str:
        text = str(text).lower()
        desc = ""
        if "double disc" in text or "2 x" in text: desc = "جفت دیسک"
        elif "single disc" in text or "disc" in text: desc = "تک دیسک"
        elif "drum" in text or "expanding" in text: desc = "کاسه‌ای"
        
        size_match = re.search(r'([\d\.]+)\s*mm', text)
        if desc and size_match:
            return self._to_persian_digits(f"{desc} {size_match.group(1)} میلی‌متری")
        if desc: return desc
        return self._to_persian_digits(text)

    def _parse_simple_metric(self, text: str, unit: str) -> str:
        val = self._extract_metric_value(text, unit)
        return self._to_persian_digits(val)

    def _parse_tire(self, text: str) -> str:
        match = re.search(r'(\d+/\d+[A-Z]*[R-]*\d+[A-Z\d/]*)', str(text).upper().replace(" ", ""))
        if match:
            return self._to_persian_digits(match.group(1))
        return self._to_persian_digits(text)

    def _parse_bore_stroke(self, text: str) -> str:
        match = re.search(r'([\d\.]+)\s*(?:x|X|×|\*)\s*([\d\.]+)', str(text))
        if match:
            formatted_val = f"{match.group(1)} × {match.group(2)} میلی‌متر"
            return self._to_persian_digits(formatted_val)
        return self._to_persian_digits(text)

    def _parse_suspension(self, text: str) -> str:
        text = re.sub(r'([\d\.]+)\s*mm', r'\1 میلی‌متر', str(text), flags=re.IGNORECASE)
        suspension_dict = [
            (r'\binverted fork(s)?\b', 'دوشاخ معکوس (USD)'),
            (r'\bmonocross\b', 'کمک‌فنر مرکزی (موناکراس)'),
            (r'\binverted\b', 'معکوس (USD)'),
            (r'\bsingle shock\b', 'تک کمک‌فنر'),
            (r'\bsingle\b', 'تک'),
            (r'\btelescopic fork(s)?\b', 'دوشاخ تلسکوپی'),
            (r'\bfork(s)?\b', 'دوشاخ'),
            (r'\b(usd|upside down|upside-down)\b', 'معکوس (USD)'),
            (r'\bmonoshock dampers?\b', 'کمک‌فنر مرکزی (مونوشوک)'),
            (r'\bmonoshock\b', 'کمک‌فنر مرکزی (مونوشوک)'),
            (r'\bhydraulic dampers?\b', 'کمک‌فنر هیدرولیکی'),
            (r'\bhydraulic\b', 'هیدرولیکی'),
            (r'\bdampers?\b', 'کمک‌ فنر'),
            (r'\bshocks?\b', 'کمک‌ فنر'),
            (r'\bswingarm|swing arm\b', 'دوشاخ عقب'),
            (r'\bcoil spring\b', 'فنر لول'),
            (r'\bsprings?\b', 'فنر'),
            (r'\bfully adjustable\b', 'کاملاً قابل تنظیم'),
            (r'\bstep adjustable\b', 'مرحله‌ای قابل تنظیم'),
            (r'\badjustable\b', 'قابل تنظیم'),
            (r'\bsingle arm\b', 'تک بازو'),
            (r'\bdual action\b', 'عملکرد دوگانه'),
            (r'\blink suspension\b', 'سیستم تعلیق لینک‌دار'),
            (r'\bspeed sensitive\b', 'حساس به سرعت'),
            (r'\bdamping\b', 'میرایی (Damping)'),
            (r'\bwith\b', 'با'),
            (r'\band\b', 'و'),
            (r'\btype\b', 'نوع'),
            (r'\bdual\b', 'جفت'),
            (r'\btwin\b', 'جفت'),
            (r'\btelescopic\b', 'تلسکوپی'),
        ]
        
        for eng, per in suspension_dict:
            text = re.sub(eng, per, text, flags=re.IGNORECASE)
            
        text = text.rstrip(', ')
        text = text.replace('معکوس (معکوس (USD))', 'معکوس (USD)')
        text = text.replace('دوشاخ معکوس (USD) (USD)', 'دوشاخ معکوس (USD)')
        return self._to_persian_digits(text)

    def _parse_wheels(self, text: str) -> str:
        text = str(text).lower()
        translations = {
            r'\bsix spoke\b': 'شش پره',
            r'\bfive spoke\b': 'پنج پره',
            r'\bthree spoke\b': 'سه پره',
            r'\bten spoke\b': 'ده پره',
            r'\bspoke(s)?\b': 'پره‌ای (سیمی)',
            r'\blaced\b': 'پره‌دار',
            r'\bcast aluminium\b': 'آلومینیومی (Cast)',
            r'\bcast aluminum\b': 'آلومینیومی (Cast)',
            r'\baluminium alloy\b': 'آلیاژ آلومینیوم',
            r'\baluminum alloy\b': 'آلیاژ آلومینیوم',
            r'\balloy rims?\b': 'رینگ آلیاژی',
            r'\balloy\b': 'آلیاژی',
            r'\bcast\b': 'ریخته‌گری شده (Cast)',
            r'\baluminium\b': 'آلومینیومی',
            r'\baluminum\b': 'آلومینیومی',
            r'\btubeless\b': 'تیوب‌لس',
            r'\bsteel\b': 'فولادی'
        }
        
        for eng, per in translations.items():
            text = re.sub(eng, per, text, flags=re.IGNORECASE)
            
        return self._to_persian_digits(text.strip())
    # =========================================================
    # کنترل کننده‌های اصلی
    # =========================================================
    def flatten_dict(self, d, parent_key='', sep='_'):
        """این تابع جیسون‌های تو در تو (مثل خروجی Bikez) را کاملاً تخت می‌کند."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((k, v)) # ما فقط کلید اصلی را نگه می‌داریم تا با مپینگ مچ شود
        return dict(items)

    def _parse_crawled_model_name(self, raw_data: dict) -> str:
        """
        استخراج نام مدل کراول شده از دیتای خام (قبل از فلت شدن)
        """
        # حالت اول: سایت MCS (نام مدل مستقیماً در روت قرار دارد)
        name = raw_data.get("Make Model") or raw_data.get("Model")
        
        # حالت دوم: سایت Bikez (نام مدل داخل دیکشنری General قرار دارد)
        if not name and "General information" in raw_data and isinstance(raw_data["General information"], dict):
            name = raw_data["General information"].get("Model")
            
        return str(name).strip() if name else ""

    def parse_specs(self, raw_data: dict) -> dict:
        """پارس کردن فیچرهای مشخصات فنی از هر دو سورس"""
        if not raw_data:
            return {}
            
        # 1. اول جیسون را کاملا فلت می‌کنیم تا دیتای تو در توی Bikez حل شود
        flat_data = self.flatten_dict(raw_data)

        # 2. در Motorcyclespecs گاهی ابعاد به هم چسبیده است. آن را تفکیک می‌کنیم
        if "Dimensions" in flat_data:
            dim_text = str(flat_data["Dimensions"])
            length_match = re.search(r'Length:\s*([\d\.]+)\s*mm', dim_text, re.IGNORECASE)
            width_match = re.search(r'Width\s*([\d\.]+)\s*mm', dim_text, re.IGNORECASE)
            height_match = re.search(r'Height\s*([\d\.]+)\s*mm', dim_text, re.IGNORECASE)
            
            if length_match: flat_data["Overall length"] = f"{length_match.group(1)} mm"
            if width_match: flat_data["Overall width"] = f"{width_match.group(1)} mm"
            if height_match: flat_data["Overall height"] = f"{height_match.group(1)} mm"

        parsed_specs = {"crawled_name": self._parse_crawled_model_name(raw_data)}
        
        for eng_std_key, source_keys in self.key_map.items():
            for ek in source_keys:
                if ek in flat_data and flat_data[ek]:
                    raw_value = self._fix_bikez_broken_text(flat_data[ek])
                    
                    if eng_std_key == "capacity": val = self._parse_capacity(raw_value)
                    elif eng_std_key == "engine_type": val = self._parse_engine_type(raw_value)
                    elif eng_std_key == "max_power": val = self._parse_power_torque(raw_value, is_power=True)
                    elif eng_std_key == "max_torque": val = self._parse_power_torque(raw_value, is_power=False)
                    elif eng_std_key == "cooling_system": val = self._parse_cooling(raw_value)
                    elif eng_std_key == "fuel_system": val = self._parse_fuel_system(raw_value)
                    elif eng_std_key == "starter": val = self._parse_starter(raw_value)
                    elif eng_std_key == "transmission": val = self._parse_transmission(raw_value)
                    elif eng_std_key == "final_drive": val = self._parse_final_drive(raw_value)
                    elif eng_std_key == "clutch": val = self._parse_clutch(raw_value)
                    elif eng_std_key == "frame": val = self._parse_frame(raw_value)
                    elif "brakes" in eng_std_key: val = self._parse_brakes(raw_value)
                    elif "tyre" in eng_std_key: val = self._parse_tire(raw_value)
                    elif eng_std_key in ["seat_height", "ground_clearance"]: val = self._parse_simple_metric(raw_value, "میلی‌متر")
                    elif eng_std_key == "weight": val = self._parse_simple_metric(raw_value, "کیلوگرم")
                    elif eng_std_key == "fuel_capacity": val = self._parse_simple_metric(raw_value, "لیتر")
                    elif eng_std_key == "compression_ratio": val = self._to_persian_digits(raw_value)
                    elif eng_std_key == "bore_x_stroke": val = self._parse_bore_stroke(raw_value)
                    elif eng_std_key in ["front_suspension", "rear_suspension"]: val = self._parse_suspension(raw_value)
                    elif eng_std_key == "wheels": val = self._parse_wheels(raw_value)
                    else: val = self._to_persian_digits(raw_value)

                    parsed_specs[eng_std_key] = val
                    break # اگر کلید پیدا شد، جستجو در سورس کلیدها متوقف شود

        return parsed_specs

    def parse_ratings(self, raw_ratings: dict) -> dict:
        """پارس کردن دیتاهای ریتینگ"""
        if not raw_ratings:
            return {}

        parsed_ratings = {}
        
        for raw_key, std_key in self.rating_map.items():
            if raw_key in raw_ratings:
                bike_score = raw_ratings[raw_key].get("bike")
                if bike_score is not None:
                    parsed_ratings[std_key] = round(float(bike_score), 1)

        if "_voter_count" in raw_ratings:
            parsed_ratings["voter_count"] = int(raw_ratings["_voter_count"])

        return parsed_ratings
    