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
    "نیروموتور": "Niroo Motor",
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
    # CG / CGL / CD Family
    "CG 125": {
        "parent_brand": "Honda",
        "keywords": ["cg 125", "cg125", "هوندا سی جی 125", "هوندا 125", "هندا 125", "هندا ۱۲۵", "سی جی 125", "رهرو 125", "کویر 125", "ساوین 125", "احسان 125"],
        "exclusions": ["150", "۱۵۰", "200", "۲۰۰", "click", "کلیک", "cgl", "cd", "cdi", "باکسر", "boxer"],
        "copy_brands": ["Kavir", "Kabir", "Kasir", "Hamtaz", "Parvaz", "Xigma", "Ehsan", "Niroo Moharekeh", "Rika", "Hormoz", "Hirmand", "Tondar Shahab", "Pishro", "Iran Doocharkh", "Behran", "Dino", "Shahin Motor", "Savin", "Matin Khodro", "Amico", "Talash", "Zomorod Kavir", "Ashil"]
    },
    "CG 150": {
        "parent_brand": "Honda",
        "keywords": ["cg 150", "cg150", "هوندا 150", "هندا 150", "هندا ۱۵۰", "سی جی 150", "رهرو 150", "کویر 150", "استارتی 150", "احسان 150"],
        "exclusions": ["125", "۱۲۵", "200", "۲۰۰", "click", "کلیک", "cgl", "cd", "cdi", "باکسر", "boxer", "ایروکس", "aerox"],
        "copy_brands": ["Kavir", "Parvaz", "Ehsan", "Lifan", "Savin", "Kabir", "Hamtaz", "Niroo Moharekeh", "Iran Doocharkh", "Zomorod Kavir", "Ashil"]
    },
    "CG 200": {
        "parent_brand": "Honda",
        "keywords": ["cg 200", "cg200", "هوندا 200", "هندا 200", "سی جی 200"],
        "exclusions": ["125", "150", "cgl", "cd", "cdi"],
        "copy_brands": ["Kavir", "Kabir", "Kasir", "Hamtaz", "Parvaz", "Ehsan", "Pishro", "Iran Doocharkh", "Lifan"]
    },
    "CGL 125": {"parent_brand": "Honda", "keywords": ["cgl 125", "سی جی ال 125", "cgl125"], "exclusions": ["150", "200"], "copy_brands": []},
    "CGL 150": {"parent_brand": "Honda", "keywords": ["cgl 150", "سی جی ال 150", "cgl150", "glx 150"], "exclusions": ["125", "200"], "copy_brands": ["Kavir", "Kasir"]},
    "CGL 200": {"parent_brand": "Honda", "keywords": ["cgl 200", "سی جی ال 200", "cgl200"], "exclusions": ["125", "150"], "copy_brands": ["Hamtaz"]},
    "CD 125": {"parent_brand": "Honda", "keywords": ["cd 125", "cd125"], "exclusions": ["cdi", "150"], "copy_brands": ["Kavir"]},
    "CDI 125": {"parent_brand": "Honda", "keywords": ["cdi 125", "سی دی آی 125", "cdi125"], "exclusions": ["150"], "copy_brands": []},
    "CDI 150": {"parent_brand": "Honda", "keywords": ["cdi 150", "cdi150", "سی دی آی 150"], "exclusions": ["125"], "copy_brands": ["Kavir"]},

    # Click / Vario Family
    "Click 125": {"parent_brand": "Honda", "keywords": ["کلیک  125", "کلیک125", "click 125", "کیلیک 125"], "exclusions": ["150", "160", "170", "180"], "copy_brands": []},
    "Click 150": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 150", "کلیک ۱۵۰", "click 150", "کیلیک 150", "کیلیک ۱۵۰", "vario 150", "واریو 150",
                      "واریو ۱۵۰", "های کلیک", "طرح کلیک 150","کلیک 150 تایلندی"
                      ],
        "exclusions": ["160", "۱۶۰", "170", "۱۷۰", "125", "۱۲۵", "180", "۱۸۰", "adv", "ایدیلینگ 160"],
        "copy_brands": ["Kavir", "Niroo Motor", "Salar Gostar", "Takht-e-Jamshid", "Hani", "Pishro", "Saqeb", "Iran Doocharkh", "Toosan", "Kabir"]
    },
    "Click 160": {
        "parent_brand": "Honda",
        "keywords": [
            "کلیک 160", "کلیک ۱۶۰", "کلیک160", "کلیک۱۶۰",
            "click 160", "click160", 
            "کیلیک 160", "کیلیک ۱۶۰", "کیلیک160", "کیلیک۱۶۰",
            "vario 160", "vario160", "واریو 160", "واریو ۱۶۰", "واریو160", "واریو۱۶۰",
            "طرح کلیک 160", "کلیک ۱۶۰ تایلندی", 
        ],
        "exclusions": ["170", "۱۷۰", "180", "۱۸۰", "adv", "pcx", "150"],
        "copy_brands": ["Niroo Motor", "Salar Gostar", "Hani", "Toosan", "Kabir", "Hamrah Cyclet Janhan", "Jahan Tond", "Arman", "Chabookro", "Alborz", "Behro", "Tekno"] 
    },
    "Click 170": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 170", "vario 170", "واریو 170", "کیلیک 170"],
        "exclusions": ["150", "160", "180"],
        "copy_brands": ["Kavir", "Kabir"]
    },
    "Click 180": {
        "parent_brand": "Honda",
        "keywords": ["کلیک 180", "click 180", "vario 180", "واریو 180", "کیلیک 180", "واریو کبیر 180"],
        "exclusions": ["150", "160", "170"],
        "copy_brands": ["Kabir", "Hamrah Cyclet Janhan"]
    },
    "Click 165": {"parent_brand": "Honda", "keywords": ["واریو 165", "vario 165"], "exclusions": ["150", "160"], "copy_brands": []},

    # ADV Family
    "ADV 150": {
        "parent_brand": "Honda",
        "keywords": ["adv150", "adv 150", "ای دی وی 150", "ادی وی 150"],
        "exclusions": ["160", "170", "175", "350"],
        "copy_brands": ["Kavir", "Kabir", "Niroo Motor", "Nikran Motor", "Toosan", "Hani"]
    },
    "ADV 160": {
        "parent_brand": "Honda",
        "keywords": ["adv160", "adv 160", "ای دی وی 160", "ادی وی 160"],
        "exclusions": ["150", "175", "350"],
        "copy_brands": ["Hani", "Hamtaz"]
    },
    "ADV 350": {"parent_brand": "Honda", "keywords": ["adv350", "adv 350", "ای دی وی 350"], "exclusions": ["150", "160", "175"], "copy_brands": []},

    # PCX Family
    "PCX 150": {"parent_brand": "Honda", "keywords": ["pcx 150", "pcx150", "پی سی ایکس 150"], "exclusions": ["160"], "copy_brands": []},
    "PCX 160": {"parent_brand": "Honda", "keywords": ["pcx 160", "pcx160", "پی سی ایکس 160"], "exclusions": ["150"], "copy_brands": ["Salar Gostar"]},

    # Wave / Scooters
    "Wave 110": {"parent_brand": "Honda", "keywords": ["ویو 110", "wave 110", "ویو110"], "exclusions": ["125"], "copy_brands": ["Kavir", "Hani"]},
    "Wave 125": {"parent_brand": "Honda", "keywords": ["ویو 125", "ویو125", "wave 125"], "exclusions": ["110"], "copy_brands": ["Niroo Motor"]},
    "Scoopy 110": {"parent_brand": "Honda", "keywords": ["اسکوپی 110", "scoopy 110", "اسکوپی"], "exclusions": [], "copy_brands": []},
    "Activa 125": {"parent_brand": "Honda", "keywords": ["اکتیوا 125", "activa 125", "اکتیوا"], "exclusions": [], "copy_brands": []},

    # CB / CBR / CRF / Heavy Family
    "CB 1300": {"parent_brand": "Honda", "keywords": ["cb 1300", "cb1300", "سی بی 1300", "سیبی1300"], "exclusions": [], "copy_brands": []},
    "CB 1000 R": {"parent_brand": "Honda", "keywords": ["cb1000r", "cb 1000r", "cb 1000 r"], "exclusions": ["1300", "cbr"], "copy_brands": []},
    "CB 750": {"parent_brand": "Honda", "keywords": ["cb750", "cb 750", "سی بی750"], "exclusions": [], "copy_brands": []},
    "CB 400": {"parent_brand": "Honda", "keywords": ["cb 400", "cb400", "سی بی 400", "cb1 400"], "exclusions": ["cbr", "x"], "copy_brands": []},
    "CB 300": {"parent_brand": "Honda", "keywords": ["cb 300", "cb300"], "exclusions": ["cbr"], "copy_brands": []},
    "CB 250": {"parent_brand": "Honda", "keywords": ["cb 250", "cb250"], "exclusions": ["cbr", "x"], "copy_brands": []},
    "CB 200X": {"parent_brand": "Honda", "keywords": ["cb200x", "cb 200x"], "exclusions": [], "copy_brands": []},
    "CB 190": {"parent_brand": "Honda", "keywords": ["cb190", "cb 190", "cb190r", "سی بی 190"], "exclusions": [], "copy_brands": []},
    "CB 150 R": {"parent_brand": "Honda", "keywords": ["cb150r", "cb 150 r"], "exclusions": ["cbr"], "copy_brands": []},
    "CBR 150": {"parent_brand": "Honda", "keywords": ["cbr 150", "cbr150"], "exclusions": ["250", "1000"], "copy_brands": []},
    "CBR 250": {"parent_brand": "Honda", "keywords": ["cbr 250", "cbr250", "cbr250rr"], "exclusions": ["150", "1000"], "copy_brands": []},
    "CBR 300": {"parent_brand": "Honda", "keywords": ["cbr 300", "cbr300"], "exclusions": [], "copy_brands": []},
    "CBR 400": {"parent_brand": "Honda", "keywords": ["cbr 400", "cbr400"], "exclusions": [], "copy_brands": []},
    "CBR 600": {"parent_brand": "Honda", "keywords": ["cbr 600", "cbr600", "cbr600rr"], "exclusions": [], "copy_brands": []},
    "CBR 1000 RR": {"parent_brand": "Honda", "keywords": ["cbr1000rr", "cbr 1000 rr", "cbr 1000"], "exclusions": [], "copy_brands": []},
    "CRF 250": {"parent_brand": "Honda", "keywords": ["crf 250", "crf250", "crf 250r", "crfx250"], "exclusions": ["450", "rally"], "copy_brands": []},
    "CRF 450": {"parent_brand": "Honda", "keywords": ["crf 450", "crf450", "crf450r"], "exclusions": ["250"], "copy_brands": []},
    "CRF Rally 250": {"parent_brand": "Honda", "keywords": ["crf rally 250", "crf250 rally", "crf رالی 250", "رالی 250"], "exclusions": ["450"], "copy_brands": []},
    
    # SH / Forza / Other Honda
    "SH 150": {"parent_brand": "Honda", "keywords": ["sh 150", "اس اچ 150"], "exclusions": ["160", "180"], "copy_brands": []},
    "SH 160": {"parent_brand": "Honda", "keywords": ["sh 160", "اس اچ 160"], "exclusions": ["150", "180"], "copy_brands": []},
    "SH 180": {"parent_brand": "Honda", "keywords": ["sh 180", "اس اچ 180"], "exclusions": ["150", "160"], "copy_brands": ["Xigma"]},
    "Forza 250": {"parent_brand": "Honda", "keywords": ["forza 250", "فورزا 250"], "exclusions": ["350"], "copy_brands": []},
    "Forza 350": {"parent_brand": "Honda", "keywords": ["forza 350", "فورزا 350"], "exclusions": ["250"], "copy_brands": []},
    "Hornet 185": {"parent_brand": "Honda", "keywords": ["hornet 185", "هورنت 185"], "exclusions": ["200", "600"], "copy_brands": []},
    "Hornet 200": {"parent_brand": "Honda", "keywords": ["hornet 200", "هورنت 200"], "exclusions": ["185", "600"], "copy_brands": []},
    "Africa Twin 1000": {"parent_brand": "Honda", "keywords": ["africa twin 1000", "آفریقا تویین", "africa twin"], "exclusions": [], "copy_brands": []},
    "Super Cub 110": {"parent_brand": "Honda", "keywords": ["super cub 110", "سوپرکاپ 110"], "exclusions": ["90"], "copy_brands": []},
    "Gold Wing 1000": {"parent_brand": "Honda", "keywords": ["goldwing 1000", "گلدوینگ 1000"], "exclusions": ["1600"], "copy_brands": []},
    "Dio 110": {"parent_brand": "Honda", "keywords": ["dio 110", "دیو 110"], "exclusions": ["50", "125"], "copy_brands": []},
    "XR 250": {"parent_brand": "Honda", "keywords": ["xr 250", "ایکس آر 250"], "exclusions": [], "copy_brands": []},
    "XL 125": {"parent_brand": "Honda", "keywords": ["xl 125", "xl125", "ایکس ال 125"], "exclusions": ["250"], "copy_brands": []},

    # جا مانده‌های HONDA
    "CG 70": {"parent_brand": "Honda", "keywords": ["cg 70", "هوندا 70", "honda 70"], "exclusions": ["125", "150", "200"], "copy_brands": []},
    "Hornet 600": {"parent_brand": "Honda", "keywords": ["هورنت 600", "hornet 600"], "exclusions": ["185", "200"], "copy_brands": []},
    "Super Cub 90": {"parent_brand": "Honda", "keywords": ["super cub 90", "سوپرکاپ 90", "سوپر کاپ 90"], "exclusions": ["110"], "copy_brands": []},
    "Gold Wing 1600": {"parent_brand": "Honda", "keywords": ["goldwing 1600", "gold wing 1600", "گلدوینگ 1600"], "exclusions": ["1000"], "copy_brands": []},
    "Jorno": {"parent_brand": "Honda", "keywords": ["jorno", "giorno", "جیورنو"], "exclusions": [], "copy_brands": []},
    "Beat 110": {"parent_brand": "Honda", "keywords": ["beat 110", "بیت 110"], "exclusions": [], "copy_brands": []},
    "TACT": {"parent_brand": "Honda", "keywords": ["tact"], "exclusions": [], "copy_brands": []},
    "Dio 50": {"parent_brand": "Honda", "keywords": ["dio 50", "دیو 50"], "exclusions": ["110", "125"], "copy_brands": []},
    "Dio 125": {"parent_brand": "Honda", "keywords": ["dio 125", "دیو 125"], "exclusions": ["50", "110"], "copy_brands": []},
    "XLR 125": {"parent_brand": "Honda", "keywords": ["xlr 125", "ایکس ال آر 125"], "exclusions": ["250"], "copy_brands": []},
    "XLR 250": {"parent_brand": "Honda", "keywords": ["xlr 250", "ایکس ال آر 250"], "exclusions": ["125"], "copy_brands": []},
    "XL 250": {"parent_brand": "Honda", "keywords": ["xl 250", "xl250", "ایکس ال 250"], "exclusions": ["125", "xlr"], "copy_brands": []},

    # ================= YAMAHA PLATFORMS =================
    "Aerox 155": {
        "parent_brand": "Yamaha",
        "keywords": ["ایروکس 155", "aerox155", "ایروکس155", "aerox 155", "آیروکس", "ایروکس", "طرح ایروکس", "طرح آیروکس", "nvx 150", "nvx 155"],
        "exclusions": ["170", "180", "۱۸۰", "کلیک", "click"],
        "copy_brands": ["Niroo Motor", "Pishtaz", "Salar Gostar", "Takht-e-Jamshid", "Hani", "Kabir", "Xigma", "Hamtaz"]
    },
    "NMAX 155": {
        "parent_brand": "Yamaha",
        "keywords": ["nmax 155", "nmax155", "ان مکس 155", "انمکس", "ان مکس", "nmax", "n max", "nmax turbo", "techmax"],
        "exclusions": ["150", "xmax", "ایکس مکس", "ایروکس", "aerox"],
        "copy_brands": []
    },
    "Aerox 170": {
        "parent_brand": "Yamaha",
        "keywords": ["ایروکس 170", "aerox170", "aerox 170"],
        "exclusions": ["155", "180"],
        "copy_brands": ["Kabir"]
    },
    "Aerox 180": {
        "parent_brand": "Yamaha",
        "keywords": ["nvx 180", "ایروکس 180", "aerox 180"],
        "exclusions": ["155", "170"],
        "copy_brands": ["Kabir"]
    },
    "NVX 150": {
        "parent_brand": "Yamaha",
        "keywords": ["nvx 150", "ایروکس nvx 150"],
        "exclusions": ["155", "180"],
        "copy_brands": ["Kabir"]
    },
    "NMAX 150": {"parent_brand": "Yamaha", "keywords": ["nmax 150", "انمکس150"], "exclusions": ["155"], "copy_brands": []},
    "XMAX 250": {"parent_brand": "Yamaha", "keywords": ["xmax 250", "x max 250", "ایکس مکس 250", "xmax", "یاماها ایکس مکس"], "exclusions": ["300"], "copy_brands": ["Niroo Motor"]},
    "MX KING 150": {"parent_brand": "Yamaha", "keywords": ["mx king 150", "mx king", "ام ایکس کینگ", "mxking150", "mxking 150"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    
    # R / MT / Offroad
    "R15": {"parent_brand": "Yamaha", "keywords": ["r15", "r 15", "آر 15"], "exclusions": [], "copy_brands": []},
    "R25": {"parent_brand": "Yamaha", "keywords": ["r 25", "yamaha r25", "r25"], "exclusions": [], "copy_brands": []},
    "R3": {"parent_brand": "Yamaha", "keywords": ["r 3", "r3", "آر 3", "یاماها r3"], "exclusions": [], "copy_brands": []},
    "R6": {"parent_brand": "Yamaha", "keywords": ["r 6", "r6", "آر 6", "یاماها r6"], "exclusions": [], "copy_brands": []},
    "R1": {"parent_brand": "Yamaha", "keywords": ["r 1", "r1", "آر 1", "یاماها r1"], "exclusions": [], "copy_brands": []},
    "MT 15": {"parent_brand": "Yamaha", "keywords": ["mt15", "mt 15", "ام تی 15"], "exclusions": ["25", "09"], "copy_brands": []},
    "MT 25": {"parent_brand": "Yamaha", "keywords": ["mt25", "mt 25", "ام تی 25"], "exclusions": ["15", "09"], "copy_brands": []},
    "MT 09": {"parent_brand": "Yamaha", "keywords": ["mt09", "mt 9", "ام تی 9"], "exclusions": ["15", "25"], "copy_brands": []},
    "XSR 155": {"parent_brand": "Yamaha", "keywords": ["xsr155", "xsr 155", "ایکس اس ار 155"], "exclusions": [], "copy_brands": []},
    "WR 155": {"parent_brand": "Yamaha", "keywords": ["wr155", "wr 155"], "exclusions": ["250", "450"], "copy_brands": []},
    "WR 250": {"parent_brand": "Yamaha", "keywords": ["wr 250", "wr250"], "exclusions": ["155", "450"], "copy_brands": []},
    "WR 450": {"parent_brand": "Yamaha", "keywords": ["wr450", "wr 450", "wrf 450"], "exclusions": ["155", "250"], "copy_brands": []},
    "YZ 250": {"parent_brand": "Yamaha", "keywords": ["yz250", "yz 250", "yz250x", "کراس yz 250"], "exclusions": ["125", "yzf"], "copy_brands": []},
    "YZF 250": {"parent_brand": "Yamaha", "keywords": ["yzf250", "yzf 250", "yzf250f"], "exclusions": ["450"], "copy_brands": []},
    "YZF 450": {"parent_brand": "Yamaha", "keywords": ["yzf450", "yzf 450"], "exclusions": ["250"], "copy_brands": []},
    "DT 125": {"parent_brand": "Yamaha", "keywords": ["dt 125"], "exclusions": ["200"], "copy_brands": []},
    
    # Yamaha Scooters & Small
    "Lexi 155": {"parent_brand": "Yamaha", "keywords": ["lexi 155", "yamaha lexi", "لکسی"], "exclusions": ["jx", "جویمکس","galaxy","گلکسی"], "copy_brands": []},
    "Grand Filano 125": {"parent_brand": "Yamaha", "keywords": ["grand filano", "گرند فیلانو"], "exclusions": [], "copy_brands": []},
    "Fascino 125": {"parent_brand": "Yamaha", "keywords": ["fascino", "فاسکینو 125"], "exclusions": [], "copy_brands": []},
    "Mio M3": {"parent_brand": "Yamaha", "keywords": ["mio m3", "mio gear"], "exclusions": [], "copy_brands": []},
    "Vision 155": {"parent_brand": "Yamaha", "keywords": ["vision", "vision 155", "ویژن 155"], "exclusions": [], "copy_brands": []},
    "Jog": {"parent_brand": "Yamaha", "keywords": ["jog", "جوک"], "exclusions": [], "copy_brands": []},
    "Vino 50": {"parent_brand": "Yamaha", "keywords": ["وینو", "vino"], "exclusions": [], "copy_brands": []},
    "YBR 125": {"parent_brand": "Yamaha", "keywords": ["yb 125", "ybr 125", "125 یاماها"], "exclusions": [], "copy_brands": []},
    "AX 150": {"parent_brand": "Yamaha", "keywords": ["ax 150", "ax150"], "exclusions": [], "copy_brands": ["Pishtaz"]},

    # جا مانده‌های YAMAHA
    "NVX 155": {"parent_brand": "Yamaha", "keywords": ["nvx 155", "ایروکس nvx 155"], "exclusions": ["150", "180"], "copy_brands": ["Kabir"]},
    "Mio Gear 125": {"parent_brand": "Yamaha", "keywords": ["mio gear"], "exclusions": ["m3"], "copy_brands": []},
    "YZ 125": {"parent_brand": "Yamaha", "keywords": ["yz125", "yz 125"], "exclusions": ["250", "yzf"], "copy_brands": []},
    "DT 200": {"parent_brand": "Yamaha", "keywords": ["dt 200"], "exclusions": ["125"], "copy_brands": []},
    "TDR 250": {"parent_brand": "Yamaha", "keywords": ["tdr 250"], "exclusions": [], "copy_brands": []},
    "FZ1": {"parent_brand": "Yamaha", "keywords": ["fz1"], "exclusions": [], "copy_brands": []},
    "VMAX 1700": {"parent_brand": "Yamaha", "keywords": ["vmax", "vmax 1700"], "exclusions": [], "copy_brands": []},
    "XT 600": {"parent_brand": "Yamaha", "keywords": ["xt600", "xt 600"], "exclusions": [], "copy_brands": []},

    # ================= BAJAJ PLATFORMS =================
    "Pulsar 150": {
        "parent_brand": "Bajaj",
        "keywords": ["پالس 150", "pulsar 150", "as150"],
        "exclusions": ["135", "160", "180", "200", "220", "ns", "rs"],
        "copy_brands": []
    },
    "Pulsar 180": {
        "parent_brand": "Bajaj",
        "keywords": ["پالس 180", "پولسار 180", "pulsar 180", "ug4 180"],
        "exclusions": ["135", "150", "160", "200", "220", "ns", "rs"],
        "copy_brands": []
    },
    "Pulsar 200": {
        "parent_brand": "Bajaj",
        "keywords": ["پالس 200", "pulsar 200"],
        "exclusions": ["135", "150", "160", "180", "220", "ns", "rs"],
        "copy_brands": []
    },
    "Pulsar 220": {"parent_brand": "Bajaj", "keywords": ["پالس 220", "pulsar 220"], "exclusions": ["150", "180", "200"], "copy_brands": []},
    "Pulsar 135": {"parent_brand": "Bajaj", "keywords": ["پالس 135", "پالس ۱۳۵ cc"], "exclusions": ["150", "160", "180", "200", "220"], "copy_brands": []},
    "Pulsar 160": {"parent_brand": "Bajaj", "keywords": ["پالس 160"], "exclusions": ["135", "150", "180", "200", "220", "ns"], "copy_brands": []},
    "Pulsar NS 150": {"parent_brand": "Bajaj", "keywords": ["ns 150", "ns150", "ان اس 150"], "exclusions": ["160", "200", "کاوان", "na"], "copy_brands": []},
    "Pulsar NS 160": {"parent_brand": "Bajaj", "keywords": ["ns 160", "ns160", "ان اس 160"], "exclusions": ["150", "200","کاوان", "na"], "copy_brands": []},
    "Pulsar NS 200": {"parent_brand": "Bajaj", "keywords": ["ns200", "ان اس 200", "ns 200"], "exclusions": ["150", "160","کاوان", "na"], "copy_brands": []},
    "Pulsar RS 200": {"parent_brand": "Bajaj", "keywords": ["rs 200", "ار اس 200", "rs200", "آر اس 200", "پالس rs200"], "exclusions": [], "copy_brands": []},
    
    "Boxer 150": {
        "parent_brand": "Bajaj",
        "keywords": ["باکسر 150", "بوکسر 150", "boxer 150", "باکسر", "boxer"],
        "exclusions": ["125", "200", "bx", "bm", "n2"],
        "copy_brands": ["Hani"]
    },
    "Boxer 125": {"parent_brand": "Bajaj", "keywords": ["باکسر 125", "boxer 125"], "exclusions": ["150", "200"], "copy_brands": []},
    "BX 200": {"parent_brand": "Bajaj", "keywords": ["bx 200", "باکسر bx 200", "باکسر 200"], "exclusions": ["150"], "copy_brands": ["Niroo Motor", "Hani"]},
    "BM 200": {"parent_brand": "Bajaj", "keywords": ["bm 200", "باکسر bm 200"], "exclusions": ["150"], "copy_brands": ["Hani"]},
    
    "Dominar 250": {"parent_brand": "Bajaj", "keywords": ["دومینار 250", "dominar 250", "دمینار 250"], "exclusions": ["400"], "copy_brands": []},
    "Avenger 220": {"parent_brand": "Bajaj", "keywords": ["اونجر 220", "avenger 220"], "exclusions": ["200"], "copy_brands": []},
    "Vespa Bajaj (Chetak)": {"parent_brand": "Bajaj", "keywords": ["وسپا باجاج", "چیتک", "chetak"], "exclusions": [], "copy_brands": []},
    "Avenger 200": {"parent_brand": "Bajaj", "keywords": ["اونجر 200", "avenger 200"], "exclusions": ["220"], "copy_brands": []},

    # ================= TVS PLATFORMS =================
    "Apache 150": {"parent_brand": "TVS", "keywords": ["آپاچی 150", "اپاچی 150", "apache 150"], "exclusions": ["160", "180", "200"], "copy_brands": []},
    "Apache 160": {"parent_brand": "TVS", "keywords": ["آپاچی 160", "اپاچی 160", "apache 160"], "exclusions": ["150", "180", "200"], "copy_brands": []},
    "Apache 180": {"parent_brand": "TVS", "keywords": ["آپاچی 180", "اپاچی 180", "apache 180"], "exclusions": ["150", "160", "200"], "copy_brands": []},
    "Apache 200": {"parent_brand": "TVS", "keywords": ["آپاچی 200", "اپاچی 200", "apache 200"], "exclusions": ["150", "160", "180"], "copy_brands": ["Niroo Motor"]},
    "NTORQ 125": {"parent_brand": "TVS", "keywords": ["انتورک 125", "ntorq 125", "ntorq"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Jupiter 110": {"parent_brand": "TVS", "keywords": ["ژوپیتر 110", "جیوپیتر", "jupiter"], "exclusions": [], "copy_brands": []},
    "Wego 110": {"parent_brand": "TVS", "keywords": ["ویگو 110", "وگو 110", "wego 110"], "exclusions": [], "copy_brands": []},
    "Rockz 125": {"parent_brand": "TVS", "keywords": ["راکز 125", "راکـز 125", "rockz 125"], "exclusions": [], "copy_brands": []},
    "HLX 150": {
        "parent_brand": "TVS",
        "keywords": [
            "hlx 150", "hlx150", "hlx", 
            "اچ ال ایکس", "اچ ال ایکس 150", "اچ ال ایکس ۱۵۰", "اچ ال ایکس150", "اچ ال ایکس۱۵۰", 
            "tvs hlx", "tvs 150"
        ],
        "exclusions": ["160", "180", "kld", "bx", "bm"],
        "copy_brands": []
    },
    "Neo 125": {"parent_brand": "TVS", "keywords": ["نئو 125", "neo 125"], "exclusions": [], "copy_brands": []},
    "Dazz 110": {"parent_brand": "TVS", "keywords": ["داز 110", "dazz 110"], "exclusions": [], "copy_brands": []},
    "X7": {"parent_brand": "TVS", "keywords": ["x7", "تی وی اس x7"], "exclusions": [], "copy_brands": []},


    # ================= BENELLI PLATFORMS =================
    "BN 150": {"parent_brand": "Benelli", "keywords": ["bn 150", "bn150", "بنلی 150", "tnt 150", "tnt150", "tnt15"], "exclusions": ["180", "250", "300"], "copy_brands": []},
    "BN 180": {"parent_brand": "Benelli", "keywords": ["بنلی 180", "tnt 180", "tnt 180s", "180s"], "exclusions": ["150", "250", "300"], "copy_brands": []},
    "BN 250": {"parent_brand": "Benelli", "keywords": ["بنلی 250", "tnt 249", "tnt25", "tnt 250", "tnt250n"], "exclusions": ["150", "180", "300"], "copy_brands": []},
    "BN 300": {
        "parent_brand": "Benelli",
        "keywords": ["بنلی 300", "بنلی ۳۰۰", "tnt 300", "tnt300", "bn 300", "bn300", "بنلی سیصد", "بنلی300", "بنلی 300 جفت"],
        "exclusions": ["150", "180", "250", "249", "بلنتا", "trk", "leoncino"],
        "copy_brands": []
    },
    "302R": {"parent_brand": "Benelli", "keywords": ["302r"], "exclusions": [], "copy_brands": []},
    "TRK 249": {"parent_brand": "Benelli", "keywords": ["trk 249", "trk250", "بنلی trk", "ادونچر 250"], "exclusions": [], "copy_brands": []},
    "Leoncino 249": {"parent_brand": "Benelli", "keywords": ["leoncino 249", "leonchino", "لئونچینو"], "exclusions": [], "copy_brands": []},
    "Panarea 125": {"parent_brand": "Benelli", "keywords": ["panarea 125", "پانارئا 125", "بنلی پانارئا"], "exclusions": [], "copy_brands": []},
    "VZ 125": {"parent_brand": "Benelli", "keywords": ["vz 125", "بنلی vz", "وی زد 125", "vizy"], "exclusions": [], "copy_brands": []},

    # ================= KTM PLATFORMS =================
    "Duke 200": {"parent_brand": "KTM", "keywords": ["duke 200", "دوک  200"], "exclusions": ["250", "400"], "copy_brands": []},
    "Duke 250": {"parent_brand": "KTM", "keywords": ["duke 250", "دوک 250"], "exclusions": ["200", "400"], "copy_brands": []},
    "Duke 400": {"parent_brand": "KTM", "keywords": ["duke 400", "دوک 400"], "exclusions": ["200", "250"], "copy_brands": []},
    "RC 200": {"parent_brand": "KTM", "keywords": ["rc 200", "rc200"], "exclusions": ["250"], "copy_brands": []},
    "RC 250": {"parent_brand": "KTM", "keywords": ["rc 250", "rc250"], "exclusions": ["200"], "copy_brands": []},
    "EXC 250": {"parent_brand": "KTM", "keywords": ["exc 250", "exc250"], "exclusions": [], "copy_brands": []},

    # ================= KAWASAKI PLATFORMS =================
    "Ninja 250": {"parent_brand": "Kawasaki", "keywords": ["ninja 250", "نینجا 250"], "exclusions": ["300", "400", "1000"], "copy_brands": []},
    "Ninja 300": {"parent_brand": "Kawasaki", "keywords": ["ninja 300", "نینجا 300"], "exclusions": ["250", "400", "1000"], "copy_brands": []},
    "Ninja 400": {"parent_brand": "Kawasaki", "keywords": ["ninja 400", "نینجا 400"], "exclusions": ["250", "300", "1000"], "copy_brands": []},
    "Ninja 1000": {"parent_brand": "Kawasaki", "keywords": ["ninja 1000", "نینجا 1000"], "exclusions": ["250", "300", "400"], "copy_brands": []},
    "Z 250": {"parent_brand": "Kawasaki", "keywords": ["z 250", "z250"], "exclusions": ["300", "1000", "zx"], "copy_brands": []},
    "Z 300": {"parent_brand": "Kawasaki", "keywords": ["z 300", "z300"], "exclusions": ["250", "1000"], "copy_brands": []},
    "Z 1000": {"parent_brand": "Kawasaki", "keywords": ["z 1000", "z1000"], "exclusions": ["250", "300"], "copy_brands": []},
    "ZX 150": {"parent_brand": "Kawasaki", "keywords": ["zx 150", "zx150"], "exclusions": [], "copy_brands": []},
    "KLX 250": {"parent_brand": "Kawasaki", "keywords": ["klx 250", "klx250"], "exclusions": [], "copy_brands": []},
    "KX 250": {"parent_brand": "Kawasaki", "keywords": ["kx 250", "kx250"], "exclusions": [], "copy_brands": []},
    "GTO 125": {"parent_brand": "Kawasaki", "keywords": ["gto 125", "gto"], "exclusions": [], "copy_brands": []},
    "KSR 110": {"parent_brand": "Kawasaki", "keywords": ["ksr 110"], "exclusions": [], "copy_brands": []},


    # ================= SUZUKI PLATFORMS =================
    "GSX 250": {"parent_brand": "Suzuki", "keywords": ["gsx 250", "gsx250"], "exclusions": ["1000"], "copy_brands": []},
    "GSX 1000": {"parent_brand": "Suzuki", "keywords": ["gsx 1000", "gsx1000"], "exclusions": ["250"], "copy_brands": []},
    "Hayabusa 1300": {"parent_brand": "Suzuki", "keywords": ["hayabusa", "هایابوسا"], "exclusions": [], "copy_brands": []},
    "AX 100": {"parent_brand": "Suzuki", "keywords": ["ax 100", "ax100"], "exclusions": ["150"], "copy_brands": []},
    "RMX 250": {"parent_brand": "Suzuki", "keywords": ["rmx 250", "rmx250"], "exclusions": [], "copy_brands": []},
    "TSR 200": {"parent_brand": "Suzuki", "keywords": ["سوزوکی tsr 200", "tsr 200"], "exclusions": [], "copy_brands": []},


    # ================= SYM PLATFORMS =================
    "Galaxy NH 180": {
        "parent_brand": "SYM",
        "keywords": ["nh 180", "nh180", "ان اچ 180", "ان اچ ۱۸۰", "گلکسی nh", "sym nh180"],
        "exclusions": ["250", "۲۵۰", "249", "na", "ان ای", "j200", "jt200"],
        "copy_brands": ["Niroo Motor"]
    },
    "Galaxy NH 249": {"parent_brand": "SYM", "keywords": ["nh 249", "گلکسی nh 249", "nh249"], "exclusions": ["180", "na","250"], "copy_brands": ["Niroo Motor"]},
    "Galaxy J 200": {
        "parent_brand": "SYM",
        "keywords": [
            "j 200", "j200", "200j", "200 j",
            "جی 200", "جی ۲۰۰", "جی200", "جی۲۰۰",
            "گلکسی j200", "گالکسی j200", "sym j200", "j200abs", "j200sym"
        ],
        "exclusions": ["jt", "جی تی", "gt", "sr", "cx", "na", "nh", "jx", "180", "250"],
        "copy_brands": ["Niroo Motor"]
    },
    "Galaxy JT 200": {
        "parent_brand": "SYM",
        "keywords": [
            "jt 200", "jt200", "200 jt",
            "جی تی 200", "جی تی ۲۰۰", "جی تی200", "جی تی۲۰۰", "جیتی 200", "جیتی ۲۰۰",
            "gt 200", "gt200", "gt ۲۰۰", "gt۲۰۰", 
            "گلکسی jt200", "گالکسی jt", "گلکسی جی تی 200",
        ],
        # نکته حیاتی: کلمه j200 از ممنوعه‌های این مدل حذف شد! (چون در آگهی می‌نویسند نسخه جدید j200)
        "exclusions": ["na", "nh", "180", "250"],
        "copy_brands": ["Niroo Motor"]
    },
    "Galaxy NA 180": {
        "parent_brand": "SYM",
        "keywords": ["na 180", "na180", "ان ای 180", "ان ای ۱۸۰", "na-180", "گلکسی na180"],
        "exclusions": ["250", "۲۵۰", "249", "nh", "ان اچ", "j200", "jt200"],
        "copy_brands": ["Niroo Motor"]
    },
    "Galaxy NA 250": {
        "parent_brand": "SYM",
        "keywords": ["na 250", "na250", "ان ای 250", "ان ای ۲۵۰", "na 249", "na249", "na249cc"],
        "exclusions": ["180", "۱۸۰", "nh", "ان اچ", "j200", "jt200"],
        "copy_brands": ["Niroo Motor"]
    },
    "Galaxy SR 200": {"parent_brand": "SYM", "keywords": ["sr 200 گلگسی", "sym sr200", "sr 200", "sr200", "گلکسی SR200"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Galaxy CL 150": {"parent_brand": "SYM", "keywords": ["cl150", "cl 150", "گلگسی cl 150"], "exclusions": ["160", "170"], "copy_brands": ["Niroo Motor"]},
    "Galaxy CL 160": {"parent_brand": "SYM", "keywords": ["cl160", "cl 160", "گلگسی cl 160"], "exclusions": ["150", "170"], "copy_brands": ["Niroo Motor"]},
    "Galaxy CL 170": {"parent_brand": "SYM", "keywords": ["cl170", "cl 170", "گلگسی cl 170"], "exclusions": ["150", "160"], "copy_brands": []},
    "Galaxy CX 180": {"parent_brand": "SYM", "keywords": ["cx180", "cx 180", "galaxy cx180"], "exclusions": ["200"], "copy_brands": ["Niroo Motor"]},
    "Galaxy CX 200": {"parent_brand": "SYM", "keywords": ["cx200", "cx 200", "galaxy cx200"], "exclusions": ["180"], "copy_brands": []},
    "Galaxy R 155": {
        "parent_brand": "SYM",
        "keywords": ["r155", "r 155", "آر 155", "آر 155", "گلکسی r155", "گلگسی r 155"],
        "exclusions": ["ایروکس", "aerox"], 
        "copy_brands": ["Niroo Motor"]
    },
    "Galaxy OR 125": {"parent_brand": "SYM", "keywords": ["or125", "or 125"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Galaxy SF 180": {"parent_brand": "SYM", "keywords": ["sf180", "sf 180"], "exclusions": [], "copy_brands": []},
    "Galaxy FX 150": {"parent_brand": "SYM", "keywords": ["fx150", "fx 150"], "exclusions": [], "copy_brands": []},
    "Galaxy TN 110": {"parent_brand": "SYM", "keywords": ["tn110", "tn 110"], "exclusions": [], "copy_brands": []},
    "Galaxy JR 300": {"parent_brand": "SYM", "keywords": ["jr300", "jr 300"], "exclusions": [], "copy_brands": []},
    "Galaxy JX 249": {"parent_brand": "SYM", "keywords": ["jx249", "jx 249"], "exclusions": [], "copy_brands": []},
    "Galaxy k3 130": {"parent_brand": "SYM", "keywords": ["k3 130", "k3"], "exclusions": [], "copy_brands": ["Niroo Motor"]},
    "Fiddle 3": {"parent_brand": "SYM", "keywords": ["fiddle 3", "فیدل 3", "fiddle3"], "exclusions": ["4"], "copy_brands": []},
    "Fiddle 4": {"parent_brand": "SYM", "keywords": ["fiddle 4", "فیدل 4", "fiddle4"], "exclusions": ["3"], "copy_brands": []},
    "ADV(Husky) 175": {"parent_brand": "SYM", "keywords": ["adv 175", "adv175", "هاسکی 175", "husky adv175", "Husky 175", "هاسکی SYM175"], "exclusions": ["150", "160", "350","لسکی"], "copy_brands": []},
    "Lucky 185": {"parent_brand": "SYM", "keywords": ["lucky 185", "لاکی 185"], "exclusions": ["180", "200", "250"], "copy_brands": []},
    "Lucky 180": {"parent_brand": "SYM", "keywords": ["lucky 180", "لاکی 180"], "exclusions": ["185", "200", "250"], "copy_brands": []},
    "Lucky 200": {"parent_brand": "SYM", "keywords": ["lucky 200", "لاکی 200"], "exclusions": ["185", "180", "250"], "copy_brands": []},
    "Lucky 250": {"parent_brand": "SYM", "keywords": ["lucky 250", "لاکی 250"], "exclusions": ["185", "180", "200"], "copy_brands": []},
    "Joymax 250": {"parent_brand": "SYM", "keywords": ["joymax 250", "جوی مکس 250"], "exclusions": ["300"], "copy_brands": []},
    "Joymax 300": {"parent_brand": "SYM", "keywords": ["joymax 300", "جوی مکس 300"], "exclusions": ["250"], "copy_brands": []},
    "CLB 125": {"parent_brand": "SYM", "keywords": ["clb 125", "clb125"], "exclusions": [], "copy_brands": []},

    # ================= VESPA / PIAGGIO PLATFORMS =================
    "Vespa 125": {"parent_brand": "Vespa", "keywords": ["vespa 125", "وسپا 125"], "exclusions": ["150", "200", "250", "300"], "copy_brands": []},
    "Primavera 150": {
        "parent_brand": "Vespa",
        "keywords": ["primavera 150", "پریماورا 150", "وسپا پریماورا", "primavera s", "primavera"],
        "exclusions": [],
        "copy_brands": ["Niroo Motor", "Nikran Motor", "Jahanroo"]
    },
    "Sprint 125": {"parent_brand": "Vespa", "keywords": ["sprint 125", "اسپرینت 125"], "exclusions": ["150"], "copy_brands": []},
    "Sprint 150": {"parent_brand": "Vespa", "keywords": ["sprint 150", "اسپرینت 150", "sprint"], "exclusions": ["125"], "copy_brands": []},
    "VXL 150": {"parent_brand": "Vespa", "keywords": ["vxl 150", "vxl150", "وسپا 150", "وسپا vxl", "وسپا وی ایکس ال"], "exclusions": [], "copy_brands": []},
    "Vespa 200": {"parent_brand": "Vespa", "keywords": ["vespa 200", "وسپا 200", "px 200"], "exclusions": ["150", "250", "300"], "copy_brands": []},
    "LX 150": {"parent_brand": "Vespa", "keywords": ["ال ایکس 150", "vespa lx 150", "lx 150"], "exclusions": [], "copy_brands": []},
    "GTS 250": {"parent_brand": "Vespa", "keywords": ["gts 250", "جی تی اس 250", "gts 250 kls", "gts 250 super"], "exclusions": ["300", "280"], "copy_brands": []},
    "GTS 280": {"parent_brand": "Vespa", "keywords": ["gts 280", "۲۸۰ سوپر اسپرت"], "exclusions": ["250", "300"], "copy_brands": []},
    "GTS 300": {"parent_brand": "Vespa", "keywords": ["gts 300", "جی تی اس 300", "gts300", "vespa 300"], "exclusions": ["250", "280"], "copy_brands": []},
    "GTV 300": {"parent_brand": "Vespa", "keywords": ["gtv 300", "gtv300", "vespa gtv"], "exclusions": [], "copy_brands": []},
    "Vespa 946": {"parent_brand": "Vespa", "keywords": ["946", "vespa 946", "رد 946"], "exclusions": [], "copy_brands": []},
    "Piaggio Liberty 150": {"parent_brand": "Piaggio", "keywords": ["liberty 150", "پیاجیو لیبرتی", "وسپا پیاجیو 150"], "exclusions": [], "copy_brands": []},

    # ================= CFMOTO & KEEWAY & OTHERS =================
    "150 NK": {"parent_brand": "CFMOTO", "keywords": ["nk 150", "cf naked 150", "150 nk", "cfmoto nk 150"], "exclusions": ["250"], "copy_brands": []},
    "250 NK": {"parent_brand": "CFMOTO", "keywords": ["nk 250", "250 nk", "cf 250 nk"], "exclusions": ["150"], "copy_brands": []},
    "SRK 180": {"parent_brand": "CFMOTO", "keywords": ["srk180", "srk 180"], "exclusions": ["250"], "copy_brands": []},
    "SRV 200": {"parent_brand": "CFMOTO", "keywords": ["srv 200", "srv200"], "exclusions": [], "copy_brands": ["Kavir"]},
    "QJ 150": {"parent_brand": "CFMOTO", "keywords": ["qj 150", "کیوجی 150", "کیو جی 150", "qj150"], "exclusions": ["250", "nx", "ltr"], "copy_brands": ["Kavir"]},
    "QJ 250": {"parent_brand": "CFMOTO", "keywords": ["qj 250", "کیوجی 250", "srk 250"], "exclusions": ["150"], "copy_brands": ["Kavir"]},
    "QJ NX 150": {"parent_brand": "CFMOTO", "keywords": ["nx150", "qj nx150", "کیوجی nx150"], "exclusions": [], "copy_brands": []},
    "QJ LTR 150": {"parent_brand": "CFMOTO", "keywords": ["ltr 150", "qj ltr 150", "ال تی آر 150"], "exclusions": [], "copy_brands": ["Kavir"]},
    "QJ MTX 175": {"parent_brand": "CFMOTO", "keywords": ["mtx 175", "qj mtx 175"], "exclusions": [], "copy_brands": []},
    
    "K249R": {"parent_brand": "Keeway", "keywords": ["k249r"], "exclusions": ["k249n"], "copy_brands": []},
    "K249N": {"parent_brand": "Keeway", "keywords": ["k249n", "کی وی 249n"], "exclusions": ["k249r"], "copy_brands": []},
    "CT 150": {"parent_brand": "Keeway", "keywords": ["ct 150", "سی تی 150"], "exclusions": [], "copy_brands": []},
    "Viste 250": {"parent_brand": "Keeway", "keywords": ["viste 250", "کی وی ویسته 250", "ویسته 250"], "exclusions": ["k249r"], "copy_brands": []},

    "Aquila 250": {"parent_brand": "Hyosung", "keywords": ["هیوسانگ اکوییلا 250", "اکوییلا 250"], "exclusions": [], "copy_brands": []},

    "250 R": {"parent_brand": "Zontes", "keywords": ["zontes 250 r", "زونتس 250r", "r250"], "exclusions": ["n2", "s", "qj"], "copy_brands": []},
    "250 S": {"parent_brand": "Zontes", "keywords": ["zontes 250 s", "زونتس 250s", "s250"], "exclusions": ["n2", "r", "qj"], "copy_brands": []},
    "G1 200": {"parent_brand": "Zontes", "keywords": ["g1 200", "zontes g1"], "exclusions": [], "copy_brands": []},
    "N2 230": {
        "parent_brand": "Zontes",
        "keywords": ["زونتس 230", "زونتس ۲۳۰", "n2 230", "zontes n2", "n2", "زونتسn2", "زونتس 230n2"],
        "exclusions": ["250", "۲۵۰", "249", "v", "r", "s", "گالکسی", "adv", "boxer"],
        "copy_brands": ["Kavir"]
    },
    # ================= LIFAN PLATFORMS =================
    "Lifan 125": {"parent_brand": "Lifan", "keywords": ["لیفان 125", "lifan 125"], "exclusions": ["150", "200"], "copy_brands": []},
    "Lifan 150": {"parent_brand": "Lifan", "keywords": ["لیفان 150", "lifan 150"], "exclusions": ["125", "200", "kpv"], "copy_brands": []},
    "Lifan 200": {"parent_brand": "Lifan", "keywords": ["لیفان 200", "lifan 200"], "exclusions": ["125", "150", "kps", "mkz"], "copy_brands": []},
    "KPS 200": {"parent_brand": "Lifan", "keywords": ["kps 200", "کی پی اس 200"], "exclusions": [], "copy_brands": []},
    "MKZ 200": {"parent_brand": "Lifan", "keywords": ["mkz 200", "ام کی زد 200"], "exclusions": [], "copy_brands": []},
    "KPV 150": {"parent_brand": "Lifan", "keywords": ["لیفان kpv", "kpv 150"], "exclusions": [], "copy_brands": []},

    # ================= EXCLUSIVE IRANIAN PLATFORMS =================
    # مدل‌هایی که مستقیماً متعلق به یک شرکت ایرانی هستند (والد خارجی معروفی ندارند)
    "S1": {"parent_brand": "Kavir", "keywords": ["کویر s1", "kavir s1"], "exclusions": ["s2", "s4", "s5", "s7"], "copy_brands": []},
    "S2": {"parent_brand": "Kavir", "keywords": ["s2 150", "s2 170", "کویر s2", "s2 170i"], "exclusions": ["s1", "s4", "s5", "s7"], "copy_brands": []},
    "S4 150": {"parent_brand": "Kavir", "keywords": ["s4 150", "کویر s4"], "exclusions": ["s1", "s2", "s5", "s7"], "copy_brands": []},
    "S5 150": {"parent_brand": "Kavir", "keywords": ["s5 150", "کویر s5", "طرح وسپا کویر۱۵۰ s5"], "exclusions": ["s1", "s2", "s4", "s7"], "copy_brands": []},
    "S7 170": {"parent_brand": "Kavir", "keywords": ["s7 170", "s170", "s7-170", "s7 170i", "ایروکس کویر", "طرح آیروکس کویر"], "exclusions": ["s1", "s2", "s4", "s5"], "copy_brands": []},
    "C2": {"parent_brand": "Kavir", "keywords": ["کویـر c2", "kavir c2", "c2 250"], "exclusions": ["c3"], "copy_brands": []},
    "C3": {"parent_brand": "Kavir", "keywords": ["c3 کویر", "c3 250", "kavir c3"], "exclusions": ["c2"], "copy_brands": []},
    "N6": {"parent_brand": "Kavir", "keywords": ["n6 190", "n6 200", "کویر n6"], "exclusions": [], "copy_brands": []},
    "MS1 50": {"parent_brand": "Kavir", "keywords": ["ms1 50", "کویر ms1 50"], "exclusions": [], "copy_brands": []},
    "AGV 150": {"parent_brand": "Kavir", "keywords": ["agv 150", "agv150", "agv x", "ای جی وی 150"], "exclusions": [], "copy_brands": []},
    "RKV 200": {"parent_brand": "Kavir", "keywords": ["rkv 200"], "exclusions": [], "copy_brands": []},
    "Trail T2/T4": {"parent_brand": "Kavir", "keywords": ["تریل t2", "تریل t4", "trail 200", "trail 250"], "exclusions": [], "copy_brands": []},

    "KLD 180": {"parent_brand": "Kabir", "keywords": ["kld 180"], "exclusions": ["200"], "copy_brands": []},
    "KLD 200": {
            "parent_brand": "Kabir", # KLD is originally Kabir's Boxer copy
            "keywords": ["kld 200", "kld200", "kld", "کی ال دی 200", "کی ال دی", "کی ال دی ۲۰۰"],
            "exclusions": ["180", "۱۸۰", "bx", "bm", "hlx", "150", "کویر"],
            "copy_brands": []
    },
    "Savin 125": {"parent_brand": "Savin", "keywords": ["ساوین 125", "savin 125"], "exclusions": ["150"], "copy_brands": []},
    "Savin 150": {"parent_brand": "Savin", "keywords": ["ساوین 150", "savin 150"], "exclusions": ["125"], "copy_brands": []},

    "Hamtaz 125": {"parent_brand": "Hamtaz", "keywords": ["همتاز 125"], "exclusions": ["150", "200"], "copy_brands": []},
    "Hamtaz 150": {"parent_brand": "Hamtaz", "keywords": ["همتاز 150", "طرح وسپا همتاز 150"], "exclusions": ["125", "200"], "copy_brands": []},
    "XTM 200": {"parent_brand": "Hamtaz", "keywords": ["xtm 200", "همتاز xtm", "طرح باکسر xtm"], "exclusions": [], "copy_brands": []},
    "Blenta RT2 250": {"parent_brand": "Hamtaz", "keywords": ["بلنتا rt2 250", "blenta rt2"], "exclusions": [], "copy_brands": []},
    "Blenta VLX 170": {"parent_brand": "Hamtaz", "keywords": ["بلنتا vlx 170", "blenta vlx"], "exclusions": [], "copy_brands": []},
    "Blenta Z1 249": {"parent_brand": "Hamtaz", "keywords": ["بلنتا z1 249", "blenta z1"], "exclusions": [], "copy_brands": []},

    "Daichi 125": {"parent_brand": "Daichi", "keywords": ["دایچی 125", "daichi 125"], "exclusions": ["150", "200"], "copy_brands": []},
    "Daichi 150": {"parent_brand": "Daichi", "keywords": ["دایچی 150", "daichi 150"], "exclusions": ["125", "200", "jp"], "copy_brands": []},
    "Daichi 200": {"parent_brand": "Daichi", "keywords": ["دایچی 200", "daichi 200"], "exclusions": ["125", "150"], "copy_brands": []},

    "Megelli 200": {"parent_brand": "Megelli", "keywords": ["مگلی 200", "megelli 200"], "exclusions": [], "copy_brands": []},
    "Delta 170": {"parent_brand": "Shahin Motor", "keywords": ["دلتا 170", "delta 170"], "exclusions": [], "copy_brands": []},
    "Dino 125": {"parent_brand": "Dino", "keywords": ["دینو ۱۲۵", "dino 125"], "exclusions": [], "copy_brands": []},
    "Dy 125": {"parent_brand": "Ashil", "keywords": ["آشیل 125", "dy 125", "دی وای"], "exclusions": [], "copy_brands": []},
}
