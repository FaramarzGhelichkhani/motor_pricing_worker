# ==========================================
# 1. Brands Definition
# ==========================================
ORIGINAL_BRANDS = [
    "Honda", "Yamaha", "Suzuki", "Kawasaki", "Bajaj", "TVS", "Benelli", "KTM", 
    "Vespa", "CFMOTO", "Zontes", "Lifan", "Keeway", "SYM", "GasGas", "Aprilia", 
    "Ducati", "BMW", "Daichi", "Harley-Davidson", "Hyosung", "Hero", "Kymco", 
    "Mahindra", "Kayo", "Piaggio", "Megelli", "Super Soco", "Triumph", "Daelim"
]

IRANIAN_BRANDS = [
    "Kavir", "Niroo Motor", "Kabir", "Hamtaz", "Savin", "Dino", "Kasir", 
    "Pishro", "Pishtaz", "Parvaz", "Iran Doocharkh", "Toosan", "Matin Khodro", 
    "Jahan Hamta", "Zomorod Kavir", "Hani", "Xigma", "Salar Gostar", 
    "Nikran Motor", "Jahanroo", "Takht-e-Jamshid", "Shahin Motor", "Mehran", 
    "Amico", "Ehsan", "Talash", "Shabab", "Tondar Shahab", "Hormoz", "Rika", "Saqeb"
]

# =========================================================
# BRAND MAP  (alias -> canonical brand)
# =========================================================
BRAND_NAME_MAP = {
    "honda": "Honda", "هوندا": "Honda", "yamaha": "Yamaha", "یاماها": "Yamaha", "suzuki": "Suzuki", "سوزوکی": "Suzuki","kawasaki": "Kawasaki",
    "کاوازاکی": "Kawasaki", "bajaj": "Bajaj", "باجاج": "Bajaj", "بوکسر": "Bajaj", "باکسر": "Bajaj", "tvs": "TVS", "تی وی اس": "TVS",
    "اپاچی": "TVS", "آپاچی": "TVS", "hero": "Hero", "هیرو": "Hero", "ماهیندرا": "Mahindra",
    "ktm": "KTM", "کی تی ام": "KTM", "bmw": "BMW", "بی ام و": "BMW",
    "هارلی": "Harley-Davidson", "هارلی دیویدسون": "Harley-Davidson",  "تریومف": "Triumph", "دوکاتی": "Ducati",
    "vespa": "Vespa",  "وسپا": "Vespa",
    "پیاجیو": "Piaggio",
    "پیاجینو": "Piaggio",
    "آپریلیا": "Aprilia",
    "benelli": "Benelli",  "بنلی": "Benelli",
    "زونتس": "Zontes",
    "لیفان": "Lifan",
    "کایو": "Kayo",
    "کیمکو": "Kymco",
    "دایلیم": "Daelim",
    "هیوسانگ": "Hyosung",
    "گس گس": "GasGas",
    "کی وی": "Keeway",
    "keeway": "Keeway",
    "cfmoto": "CFMOTO",
    "سی اف موتو": "CFMOTO",
    "sym": "SYM",
    "اس وای ام": "SYM",
    "گلکسی": "SYM",
    # Iranian / Local
    "کویر": "Kavir",
    "کویر موتور": "Kavir",
    "کبیر": "Kabir",
    "کثیر ( رهرو )": "Kasir",
    "هانی" : "Hani",
    "همتاز": "Hamtaz",
    "ساوین": "Savin",
    "پیشرو": "Pishro",
    "پیشتاز": "Pishtaz",
    "پرواز": "Parvaz",
    "ایران دوچرخ": "Iran Doocharkh",
    "نیرو محرکه": "Niroo Moharekeh",
    "توسن": "Toosan",
    "توسن محرکه": "Toosan",
    "متین خودرو": "Matin Khodro",
    "جهان رو": "Jahanroo",
    "جهان همتا": "Jahan Hamta",
    "زمرد کویر": "Zomorod Kavir",
    "ثامن سیکلت": "Samen",
    "ثاقب خودرو": "Saqeb",
    "دلتا": "Shahin Motor",
    "شاهین موتور": "Shahin Motor",
    "احسان": "Ehsan",
    "بهران": "Behran",
    "تلاش": "Talash",
    "شباب": "Shabab",
    "مهران": "Mehran",
    "هرمز": "Hormoz",
    "زیگما": "Xigma",
    "ریکا": "Rika",
    "دایچی": "Daichi",
    "مگلی": "Megelli",
    "آمیکو": "Amico",
    "راپیدو": "Rapido",
    "سپند": "Sepand",
    "تندر شهاب": "Tondar Shahab",
    "همراه سیکلت جهان" : "Hamrah Cyclet Janhan",
    "نیرو موتور": "Niroo Motor",
    "سالار گستر":"Salar Gostar",
    "نیکران": "Nikran Motor",
    "جهان رو": "Jahanroo",
    "تخت جمشید": "Takht-e-Jamshid",
    "دینو": "Dino",
    "هیرمند":"Hirmand",
    "آشیل":"Ashil",
    "اشیل":"Ashil",
    "مادو": "Mado",
    "سوکو": "Super Soco",
}

# ==========================================
# 3. Global Platform Map
# ==========================================
GLOBAL_PLATFORM_MAP = {
    # ================= HONDA PLATFORMS =================
    "CG 125": {
        "parent_brand": "Honda",
        "keywords": ["cg 125", "cg125", "هوندا سی جی 125", "هوندا 125", "هندا 125", "سی جی 125", "رهرو", "تیزرو"],
        "exclusions": ["150", "200", "click", "کلیک"],
        "copy_brands": ["Kavir", "Kabir", "Kasir", "Hamtaz", "Parvaz", "Xigma", "Ehsan", "Niroo Moharekeh", "Rika", "Hormoz", "Hirmand", "Tondar Shahab", "Pishro", "Iran Doocharkh", "Behran"]
    },
    "CG 150": {
        "parent_brand": "Honda",
        "keywords": ["cg 150", "cg150", "هوندا سی جی 150", "هوندا 150", "هندا 150"],
        "exclusions": ["125", "200", "click", "کلیک"],
        "copy_brands": ["Kavir", "Parvaz", "Ehsan"]
    },
    "CG 200": {
        "parent_brand": "Honda",
        "keywords": ["cg 200", "cg200", "cgl 200", "هوندا 200", "هندا 200"],
        "exclusions": ["125", "150"],
        "copy_brands": ["Kavir", "Kabir", "Kasir", "Hamtaz", "Parvaz", "Ehsan", "Pishro", "Iran Doocharkh"]
    },
    "Click 150": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 150", "click 150", "کلیک150", "کیلیک 150", "vario 150", "واریو 150", "واریو150", "vario 150"],
        "exclusions": ["160", "125", "170", "165", "۱۶۰", "۱۷۰", "طرح کلیک 160"],
        "copy_brands": ["Kavir", "Niroo Motor", "Salar Gostar", "Takht-e-Jamshid", "Hani", "Pishro", "Saqeb", "Iran Doocharkh"]
    },
    "Click 160": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 160", "click 160", "vario 160", "واریو 160", "واریو ۱۶۰"],
        "exclusions": ["150", "125", "170"],
        "copy_brands": ["Niroo Motor", "Salar Gostar"] 
    },
    "Click 170": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 170", "vario 170", "واریو 170", "کیلیک 170"],
        "exclusions": ["150", "160","180"],
        "copy_brands": ["Kavir", "Kabir"]
    },
    "Click 180": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 180", "vario 180", "واریو 180"],
        "exclusions": ["150", "160", "170"],
        "copy_brands": ["Kabir"]
    },
    "ADV 150": {
        "parent_brand": "Honda",
        "keywords": ["adv150", "adv 150", "ای دی وی 150", "ادی وی 150", "adv"],
        "exclusions": ["160", "170", "175", "350"],
        "copy_brands": ["Kavir", "Kabir", "Niroo Motor", "Nikran Motor", "Toosan"]
    },
    "ADV 160": {
        "parent_brand": "Honda",
        "keywords": ["adv160", "adv 160", "ای دی وی 160"],
        "exclusions": ["150", "175", "350"],
        "copy_brands": ["Hani"]
    },
    "ADV 350": {
        "parent_brand": "Honda",
        "keywords": ["adv350", "adv 350", "ای دی وی 350"],
        "exclusions": ["150", "175", "160"],
        "copy_brands": []
    },
    "Wave 110": {
        "parent_brand": "Honda",
        "keywords": ["ویو 110", "wave 110"],
        "exclusions": ["125"],
        "copy_brands": ["Kavir", "Hani"]
    },
    "Wave 125": {
        "parent_brand": "Honda",
        "keywords": ["ویو 125", "ویو125", "wave 125"],
        "exclusions": ["110"],
        "copy_brands": ["Niroo Motor"]
    },
    
    # سایر هونداهای اورجینال (بدون کپی)
    "CB 1300": {"parent_brand": "Honda", "keywords": ["cb 1300", "cb1300", "سی بی 1300"], "exclusions": [], "copy_brands": []},
    "CBR 250": {"parent_brand": "Honda", "keywords": ["cbr 250", "cbr250", "cbr250rr"], "exclusions": [], "copy_brands": []},
    "CRF 250": {"parent_brand": "Honda", "keywords": ["crf 250", "crf250", "crf 250r"], "exclusions": [], "copy_brands": []},
    "PCX 150": {"parent_brand": "Honda", "keywords": ["pcx 150", "pcx150", "پی سی ایکس 150"], "exclusions": ["160"], "copy_brands": []},
    "PCX 160": {"parent_brand": "Honda", "keywords": ["pcx 160", "pcx160", "پی سی ایکس 160"], "exclusions": ["150"], "copy_brands": ["Salar Gostar"]},

    # ================= YAMAHA PLATFORMS =================
    "Aerox 155": {
        "parent_brand": "Yamaha",
        "keywords": ["ایروکس 155", "aerox155", "155ایروکس", "aerox 155", "nvx 155"],
        "exclusions": ["170", "180"],
        "copy_brands": ["Kavir", "Niroo Motor", "Pishtaz", "Salar Gostar", "Takht-e-Jamshid", "Hani", "Kabir", "Xigma"]
    },
    "Aerox 170": {
        "parent_brand": "Yamaha",
        "keywords": ["ایروکس 170", "aerox170", "ایروکس 170", "aerox 170",],
        "exclusions": ["155", "180"],
        "copy_brands": ["Kabir"]
    },
    "NVX 150": {
        "parent_brand": "Yamaha",
        "keywords": ["nvx 150", "ایروکس nvx 150", "nvx 150 طرح"],
        "exclusions": ["155"],
        "copy_brands": ["Kabir"]
    },
    "NMAX 155": {
        "parent_brand": "Yamaha",
        "keywords": ["nmax 155", "nmax155", "ان مکس 155", "nmax"],
        "exclusions": ["150"],
        "copy_brands": []
    },
    "XMAX 250": {
        "parent_brand": "Yamaha",
        "keywords": ["xmax 250", "x max 250", "ایکس مکس 250", "xmax"],
        "exclusions": ["300"],
        "copy_brands": ["Niroo Motor"]
    },
    "MX KING 150": {
        "parent_brand": "Yamaha",
        "keywords": ["mx king", "mxking", "ام ایکس کینگ"],
        "exclusions": [],
        "copy_brands": ["Niroo Motor"]
    },
    
    # سایر یاماها (بدون کپی)
    "MT 15": {"parent_brand": "Yamaha", "keywords": ["mt15", "mt 15", "ام تی 15"], "exclusions": ["25", "09"], "copy_brands": []},
    "MT 25": {"parent_brand": "Yamaha", "keywords": ["mt25", "mt 25", "ام تی 25"], "exclusions": ["15", "09"], "copy_brands": []},
    "R25": {"parent_brand": "Yamaha", "keywords": ["r 25", "yamaha r25", "r25"], "exclusions": [], "copy_brands": []},
    "WR 155": {"parent_brand": "Yamaha", "keywords": ["wr155", "wr 155"], "exclusions": ["250", "450"], "copy_brands": []},
    "XSR 155": {"parent_brand": "Yamaha", "keywords": ["xsr155", "xsr 155", "یاماها xsr 155"], "exclusions": [], "copy_brands": []},

    
    # ================= BAJAJ PLATFORMS =================
    "Boxer 150": {
        "parent_brand": "Bajaj",
        "keywords": ["باکسر 150", "بوکسر 150", "boxer 150", "باکسر", "boxer", "kld", "hlx"],
        "exclusions": ["125", "200"],
        "copy_brands": ["Hani"]
    },
    "BX 200": {
        "parent_brand": "Bajaj",
        "keywords": ["bx 200", "باکسر bx 200", "باکسر 200"],
        "exclusions": ["150"],
        "copy_brands": ["Niroo Motor", "Hani"]
    },
    "Pulsar NS 200": {
        "parent_brand": "Bajaj",
        "keywords": ["ns200", "ان اس 200", "ns 200"],
        "exclusions": ["150", "160"],
        "copy_brands": []
    },
    "Pulsar RS 200": {
        "parent_brand": "Bajaj",
        "keywords": ["rs 200", "ار اس 200", "rs200", "آر اس 200"],
        "exclusions": [],
        "copy_brands": []
    },
    "Pulsar 180": {
        "parent_brand": "Bajaj",
        "keywords": ["پالس 180", "پولسار 180", "pulsar 180"],
        "exclusions": ["200", "150", "220"],
        "copy_brands": []
    },

    # ================= VESPA PLATFORMS =================
    "Primavera 150": {
        "parent_brand": "Vespa",
        "keywords": ["primavera 150", "پریماورا 150", "وسپا پریماورا", "primavera"],
        "exclusions": [],
        "copy_brands": ["Niroo Motor", "Nikran Motor", "Jahanroo"]
    },
    "Sprint 150": {
        "parent_brand": "Vespa",
        "keywords": ["sprint 150", "اسپرینت 150", "sprint"],
        "exclusions": ["125"],
        "copy_brands": []
    },
    "VXL 150": {
        "parent_brand": "Vespa",
        "keywords": ["vxl 150", "vxl150", "وسپا 150", "وسپا vxl"],
        "exclusions": [],
        "copy_brands": []
    },
    "GTS 250": {
        "parent_brand": "Vespa",
        "keywords": ["gts 250", "جی تی اس 250"],
        "exclusions": ["300", "280"],
        "copy_brands": []
    },

    # ================= SYM PLATFORMS =================
    "Galaxy NA 180": {"parent_brand": "SYM", "keywords": ["na 180", "na180", "گلکسی na180"], "exclusions": ["250"], "copy_brands": ["Niroo Motor"]},
    "Galaxy NA 250": {"parent_brand": "SYM", "keywords": ["na 250", "na250", "گلکسی 250"], "exclusions": ["180"], "copy_brands": []},
    "Galaxy J 200": {"parent_brand": "SYM", "keywords": ["galaxy j 200", "گلکسی j 200", "j200", "j 200"], "exclusions": ["jt"], "copy_brands": []},
    "Galaxy SR 200": {"parent_brand": "SYM", "keywords": ["sr 200 گلگسی", "sym sr200", "sr 200"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Galaxy CL 150": {"parent_brand": "SYM", "keywords": ["cl150", "cl 150", "گلگسی cl 150", "Galaxy CL 150"], "exclusions": ["160", "170"], "copy_brands": ["Niroo Motor"]},
    "Galaxy CL 160": {"parent_brand": "SYM", "keywords": ["cl160", "cl 160", "گلگسی cl 160"], "exclusions": ["150", "170"], "copy_brands": ["Niroo Motor"]},
    "Galaxy CX 180": {"parent_brand": "SYM", "keywords": ["cx180", "cx 180", "galaxy cx180"], "exclusions": ["200"], "copy_brands": ["Niroo Motor"]},
    "Galaxy R 155": {"parent_brand": "SYM", "keywords": ["گلکسی r155", "galaxy r155"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Galaxy OR 125": {"parent_brand": "SYM", "keywords": ["or125"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Galaxy NH 249": {"parent_brand": "SYM", "keywords": ["nh 249", "گلکسی nh 249"], "exclusions": ["180"], "copy_brands": ["Niroo Motor"]},
    "Fiddle 3": {"parent_brand": "SYM", "keywords": ["fiddle 3", "فیدل 3", "فیدل3"], "exclusions": ["4"], "copy_brands": []},
    "Fiddle 4": {"parent_brand": "SYM", "keywords": ["fiddle 4", "فیدل 4", "فیدل ۴"], "exclusions": ["3"], "copy_brands": []},
    "ADV 175": {"parent_brand": "SYM", "keywords": ["adv 175", "adv175", "هاسکی 175", "husky adv175"], "exclusions": ["150", "160", "350"], "copy_brands": []},

    # ================= BENELLI PLATFORMS =================
    "BN 150": {"parent_brand": "Benelli", "keywords": ["bn 150", "bn150", "بنلی 150", "tnt 150", "tnt 15"], "exclusions": ["180", "250", "300"], "copy_brands": []},
    "BN 180": {"parent_brand": "Benelli", "keywords": ["بنلی 180", "tnt 180", "tnt 180s"], "exclusions": ["150", "250", "300"], "copy_brands": []},
    "BN 250": {"parent_brand": "Benelli", "keywords": ["بنلی 250", "tnt 249", "tnt25", "tnt 250"], "exclusions": ["150", "180", "300"], "copy_brands": []},
    "BN 300": {"parent_brand": "Benelli", "keywords": ["tnt 300", "بنلی 300", "302r", "TNT 300"], "exclusions": ["150", "180", "250"], "copy_brands": []},

    # ================= TVS PLATFORMS =================
    "Apache 180": {"parent_brand": "TVS", "keywords": ["آپاچی 180", "اپاچی 180", "apache 180"], "exclusions": ["150", "160", "200"], "copy_brands": []},
    "Apache 200": {"parent_brand": "TVS", "keywords": ["آپاچی 200", "اپاچی 200", "apache 200"], "exclusions": ["150", "160", "180"], "copy_brands": ["Niroo Motor"]},
    "NTORQ 125": {"parent_brand": "TVS", "keywords": ["انتورک 125", "ntorq 125", "ntorq"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Wego 110": {"parent_brand": "TVS", "keywords": ["ویگو 110", "وگو 110", "wego 110"], "exclusions": [], "copy_brands": []},
    "Rockz 125": {"parent_brand": "TVS", "keywords": ["راکز 125", "راکـز 125", "rockz 125"], "exclusions": [], "copy_brands": []},
    "HLX 150": {"parent_brand": "TVS", "keywords": ["hlx 150", "اچ ال ایکس 150", "hlx"], "exclusions": ["باکسر", "boxer"], "copy_brands": []},

    # ================= CFMOTO & KEEWAY =================
    "150 NK": {"parent_brand": "CFMOTO", "keywords": ["nk 150", "cf naked 150", "150 nk"], "exclusions": ["250"], "copy_brands": []},
    "250 NK": {"parent_brand": "CFMOTO", "keywords": ["nk 250", "250 nk", "cf 250 nk"], "exclusions": ["150"], "copy_brands": []},
    "QJ 150": {"parent_brand": "CFMOTO", "keywords": ["qj 150", "کیوجی 150", "کیو جی 150", "qj150"], "exclusions": ["250"], "copy_brands": ["Kavir"]},
    "QJ 250": {"parent_brand": "CFMOTO", "keywords": ["qj 250", "کیوجی 250", "srk 250"], "exclusions": ["150"], "copy_brands": ["Kavir"]},
    "K249N": {"parent_brand": "Keeway", "keywords": ["k249n", "کی وی 249n"], "exclusions": ["k249r"], "copy_brands": []},
    "Viste 250": {"parent_brand": "Keeway", "keywords": ["viste 250", "کی وی ویسته 250","ویسته 250"], "exclusions": ["k249r"], "copy_brands": []},

    # ================= Hyousong =================
    "Aquila 250":{"parent_brand":"Hyosung" , "keywords":["هیوسانگ اکوییلا 250", "اکوییلا 250"], "exclusions":[],  "copy_brands": []},

    # ================= Zontes =================
    "250 R": {"parent_brand":"Zontes", "keywords":["Zontes 250 R"], "exclusion":[], "copy_brands": []},

    # ================= EXCLUSIVE IRANIAN PLATFORMS =================
    # مدل‌هایی که مستقیماً متعلق به یک شرکت ایرانی هستند (والد دیگری ندارند)
    "S2": {"parent_brand": "Kavir", "keywords": ["s2 150", "s2 170", "کویر s2"], "exclusions": ["s4", "s5", "s7"], "copy_brands": []},
    "S4 150": {"parent_brand": "Kavir", "keywords": ["s4 150"], "exclusions": ["s2", "s5", "s7"], "copy_brands": []},
    "S5 150": {"parent_brand": "Kavir", "keywords": ["s5 150", "کویر s5"], "exclusions": ["s2", "s4", "s7"], "copy_brands": []},
    "S7 170": {"parent_brand": "Kavir", "keywords": ["s7 170", "s170", "s7-170"], "exclusions": ["s2", "s4", "s5"], "copy_brands": []},
    "AGV 150": {"parent_brand": "Kavir", "keywords": ["agv 150", "agv150", "agv x"], "exclusions": [], "copy_brands": []},
    "KLD 180": {"parent_brand": "Kabir", "keywords": ["kld 180"], "exclusions": ["200"], "copy_brands": []},
    "KLD 200": {"parent_brand": "Kabir", "keywords": ["kld 200", "kld200"], "exclusions": ["180"], "copy_brands": []},
    "Savin 125": {"parent_brand": "Savin", "keywords": ["ساوین 125", "savin 125"], "exclusions": ["150"], "copy_brands": []},
    "Savin 150": {"parent_brand": "Savin", "keywords": ["ساوین 150", "savin 150"], "exclusions": ["125"], "copy_brands": []},
    "Hamtaz 150": {"parent_brand": "Hamtaz", "keywords": ["همتاز 150"], "exclusions": ["125", "200"], "copy_brands": []},
    "Delta 170": {"parent_brand": "Shahin Motor", "keywords": ["دلتا 170", "delta 170"], "exclusions": [], "copy_brands": []},
}
