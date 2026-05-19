import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from core.config import GEMINI_API_KEY, MODEL_NAME

# ==========================================
# 2. Schema Definition (Actor-Critic Mode for BATCH)
# ==========================================
class MotorAnalysisResult(BaseModel):
    token: str = Field(description="شناسه آگهی (همان توکنی که در ورودی داده شد)")
    is_valid_ad: bool = Field(description="""اگر آگهی فقط مربوط به فروش 'یک عدد' موتور مشخص است true؛ در غیر این صورت (تبلیغ فروشگاه، لیست قیمت، قطعات، تعمیرات) false. آیا آگهی منطقی و واقعی است؟ (False اگر مدل/برند ادعایی وجود خارجی ندارد، مثل پالس 125)""")
    is_system_guess_correct: bool = Field(description="آیا حدس سیستم (sys_brand و sys_model) 100 درصد صحیح است؟")
    
    corrected_brand: str = Field(description="نام استاندارد برند واقعی (مثلاً Kavir, Honda). حتما انگلیسی باشد.")
    corrected_model: str = Field(description="نام استاندارد پلتفرم موتور (مثلاً Click 150, Aerox 155). سال و رنگ را نیاورید.")
    is_copy: bool = Field(description="آیا این موتور طرح/کپی ایرانی است؟ (True)")
    
    seller_type: str = Field(description="'private' (شخصی) یا 'dealer' (نمایشگاه/دلال)؟")
    technical_score: int = Field(description="نمره سلامت فنی و ظاهری از 1 تا 5")
    is_real_price: bool = Field(description="قیمت کل است (True) یا پیش‌پرداخت/قسط (False)؟")
    confidence_score: float = Field(description="میزان اطمینان به صحت این تحلیل (0.0 تا 1.0)")
    
    # Semantic Flags
    flag_clean: bool = Field(description="تاکید بر تمیزی، بی‌خط و خش بودن، در حد صفر یا عروسک؟")
    flag_accessories: bool = Field(description="دارای لوازم جانبی (طلق، باکس، هدلایت، گارد، پروتکشن)؟")
    flag_new_consumables: bool = Field(description="قطعات مصرفی (لاستیک، باتری، تسمه) به تازگی نو شده‌اند؟")
    flag_first_owner: bool = Field(description="تک برگ سند، از صفر دست خودم بوده؟")
    flag_new: bool = Field(description="موتور کاملا صفر، خشک است؟")
    flag_white_doc: bool = Field(description="سند سفید یا کاردکس؟")
    flag_full_docs: bool = Field(description="مدارک کامل، سند آماده انتقال، کارت سوخت دارد؟")
    flag_incomplete_docs: bool = Field(description="بدون مدارک، قولنامه‌ای، مزایده، گم شده؟")
    flag_insurance: bool = Field(description="بیمه دارد؟")
    flag_accident: bool = Field(description="سابقه تصادف، زمین‌خوردگی، شکستگی؟ (توجه: بدون تصادف = False)")
    flag_engine_issue: bool = Field(description="نقص فنی، روغن‌سوزی، تعمیر موتور؟ (توجه: بدون روغن‌ریزی = False)")
    flag_installment: bool = Field(description="فروش اقساطی، چکی، سفته؟")
    flag_swap: bool = Field(description="مایل به معاوضه؟")
    flag_urgent: bool = Field(description="فروش فوری، پول لازم، زیر قیمت؟")
    flag_service: bool = Field(description="تازه سرویس شده، روغن تعویض شده، آچارکشی شده؟")

class BatchAnalysisResult(BaseModel):
    results: list[MotorAnalysisResult]    

class AICritic:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def analyze_batch(self, batch_payloads):
        prompt = f"""
        شما کارشناس ارشد موتور سیکلت هستید.
        در ادامه یک آرایه JSON شامل اطلاعات {len(batch_payloads)} آگهی مجزا ارسال شده است.
        شما باید هر آگهی را تحلیل کرده و نتیجه را برای همه در قالب یک لیست بازگردانید.
        
        قوانین بسیار مهم:
        - خروجی فقط JSON و زبان مقادیر رشته‌ای باید 'انگلیسی' باشد.
        - از نام‌های کانونیکال استفاده کن (مثلاً Honda یا Kavir). ویژگی‌های رنگ و سال را در نام مدل نیاور.
        - اگر حدس سیستم صحیح است، is_system_guess_correct را True کن و همان مقادیر را در corrected_brand/model برگردان.
        - اگر حدس سیستم غلط است یا خالی است، خودت مقادیر صحیح را در corrected_brand/model بنویس.
        - اگر مدل ادعایی وجود خارجی ندارد (مثلا پالس 125)، is_valid_ad را False کن.
        - 'واریو' همان 'Click' است.
        - اگر موتور 'طرح' است، حتماً is_copy باید True باشد و برند باید نام شرکت ایرانی (مثل Kavir, Niroo Motor) باشد.
        - اگر از اطلاعات مندرج در تایتل و آگهی، برند یا مدل موتور قابل استخراج نبود، فیلد is_valid_ad را false بگذار.
        - فلاگ‌ها: نفی را بفهم! 'بدون روغن‌ریزی' یعنی flag_engine_issue=False. 'بدون تصادف' یعنی flag_accident=False.
        
        داده‌های ورودی (Batch):
        {json.dumps(batch_payloads, ensure_ascii=False)}
        """
        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchAnalysisResult,
                    temperature=0.0,
                ),
            )
            return response.parsed
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return None
