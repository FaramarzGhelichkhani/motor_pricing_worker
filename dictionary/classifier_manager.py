import re
from dictionary.mappings import GLOBAL_PLATFORM_MAP, BRAND_NAME_MAP

class EngineManager:
    def __init__(self):
        self.raw_data = GLOBAL_PLATFORM_MAP
        self._compiled_cache = {}

    def normalize_text(self, text):
        if not isinstance(text, str): return ""
        text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        text = text.lower().replace('\u200c', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def build_pattern(self, kw):
        kw_norm = self.normalize_text(kw)
        if kw_norm.isnumeric():
            return re.compile(rf'(?<!\d){kw_norm}(?!\d)')
        pattern = re.escape(kw_norm).replace(r'\ ', r'\s*')
        if re.search(r'^[a-z0-9\s]+$', kw_norm) and len(kw_norm.replace(" ", "")) <= 4:
            pattern = r'(?<![a-z0-9])' + pattern
        return re.compile(pattern)

    def get_patterns(self, platform_name, key):
        if platform_name not in self._compiled_cache:
            self._compiled_cache[platform_name] = {
                "keywords": [self.build_pattern(kw) for kw in self.raw_data[platform_name]["keywords"]],
                "exclusions": [self.build_pattern(exc) for exc in self.raw_data[platform_name]["exclusions"]]
            }
        return self._compiled_cache[platform_name][key]

    def rule_engine_guess(self, title, description, brand_model_raw):
        # 1. پیش‌پردازش
        if "معاوضه" in description: description = description.split("معاوضه")[0]
        if "معاوضه" in title: title = title.split("معاوضه")[0]
        
        full_text = self.normalize_text(f"{title} {description} {brand_model_raw}")
        
        scores = {}
        for platform_name in self.raw_data.keys():
            kw_pats = self.get_patterns(platform_name, "keywords")
            exc_pats = self.get_patterns(platform_name, "exclusions")
            
            score = 0
            if any(p.search(full_text) for p in kw_pats):
                score += 20
            if any(p.search(full_text) for p in exc_pats): 
                score -= 30
            
            if score > 0:
                scores[platform_name] = score

        if not scores: return None, None
        
        best_platform = max(scores, key=scores.get)
        p_data = self.raw_data[best_platform]
        
        # تشخیص برند کپی
        real_brand = None
        if p_data.get("copy_brands"):
            for fa_name, en_name in BRAND_NAME_MAP.items():
                if en_name in p_data["copy_brands"] and (fa_name in full_text or en_name.lower() in full_text):
                    real_brand = en_name
                    break
            if not real_brand and any(word in full_text for word in ["طرح", "کپی"]):
                 real_brand = "Unknown Iranian Brand"

        return (real_brand or p_data["parent_brand"]), best_platform

engine = EngineManager()
