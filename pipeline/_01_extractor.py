import json
from pipeline._02_cleaner import clean_engine_volume, clean_year, extract_last_update


def extract_raw_cleaned_features(json_data):
    json_data = json.loads(json_data)
    we = json_data.get("webengage", {})
    raw_dict = {
        "price": we.get("price", 0),
        "city": we.get("city", ""),
        "district": we.get("district", ""),
        "brand_model_raw": we.get("brand_model", ""),
        "title": json_data.get("share", {}).get("title", ""),
        "description": "",
        "publish_date_raw": ""
    }
    
    last_desc_title = None
    
    def process_w(w, s_name):
        nonlocal last_desc_title
        w_t = w.get("widget_type")
        d = w.get("data", {})

        if s_name == "TITLE" and w_t == "EXPANDABLE_SECTION":
            for inner in d.get("widget_list", []):
                if inner.get("widget_type") == "DESCRIPTION_ROW":
                    raw_dict["publish_date_raw"] = inner.get("data", {}).get("text", "")
        elif s_name == "DESCRIPTION" and w_t == "DESCRIPTION_ROW" and d.get("is_primary"):
            raw_dict["description"] = d.get("text", "")
        elif s_name == "LIST_DATA":
            if w_t == "GROUP_INFO_ROW":
                for item in d.get("items", []):
                    raw_dict[item.get("title", "")] = item.get("value", "")
            elif w_t == "UNEXPANDABLE_ROW":
                raw_dict[d.get("title", "")] = d.get("value", "")
            elif w_t == "DESCRIPTION_ROW":
                last_desc_title = d.get("text", "")
            elif w_t == "WRAPPER_ROW" and last_desc_title:
                chips = [chip.get("text", "") for chip in d.get("chip_list", {}).get("chips", [])]
                raw_dict[last_desc_title] = " | ".join(chips)
                last_desc_title = None
            elif w_t == "SELECTOR_ROW":
                modal = d.get("action", {}).get("payload", {}).get("modal_page", {})
                if modal:
                    for modal_w in modal.get("widget_list", []): process_w(modal_w, "LIST_DATA")

    for section in json_data.get("sections", []):
        for widget in section.get("widgets", []):
            process_w(widget, section.get("section_name"))

    return {
        "title": raw_dict["title"],
        "description": raw_dict["description"],
        "city": raw_dict["city"],
        "district": raw_dict["district"],
        "brand_model_raw": raw_dict["brand_model_raw"],
        "publish_date": extract_last_update(raw_dict.get("publish_date_raw")),
        "price": raw_dict["price"],
        
        "mileage": clean_engine_volume(raw_dict.get("کارکرد")),
        "production_year": clean_year(raw_dict.get("مدل (سال تولید)")),
        "engine_volume": clean_engine_volume(raw_dict.get("حجم موتور")),
        
        "color": raw_dict.get("رنگ", ""),
        "clutch_type": raw_dict.get("نوع کلاچ", ""),
        "brake_type": raw_dict.get("نوع ترمز", ""),
        "start_type": raw_dict.get("نوع استارت", ""),
        "engine_condition": raw_dict.get("وضعیت فنی موتور", ""),
        "body_condition": raw_dict.get("وضعیت بدنه", ""),
        "document_status": raw_dict.get("وضعیت سند و مدارک", "")
    }
