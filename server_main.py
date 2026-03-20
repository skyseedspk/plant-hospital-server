import os
import json
import re
import base64
import traceback
from typing import Any, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SUPPORTED_LANGS = {"en", "ur", "hi", "ar", "zh"}


# =========================
# Helpers
# =========================
def safe_str(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def normalize_lang(lang: str) -> str:
    lang = (lang or "ur").strip().lower()
    return lang if lang in SUPPORTED_LANGS else "ur"


def default_result(lang: str) -> Dict[str, str]:
    if lang == "en":
        return {
            "plant_name": "Unknown",
            "botanical_name": "Unknown",
            "confidence": "low",
            "leaf_condition": "Unable to determine clearly",
            "possible_issue": "Could not identify a reliable issue from the image",
            "causes": "Image may be unclear or symptoms may not be visible enough",
            "treatment_organic": "Upload a clearer image in daylight and include close symptoms",
            "treatment_chemical": "Use only locally approved products according to the label",
            "use_guidance": "Consult a local plant expert before using chemical treatment",
            "safety_notes": "Keep chemicals away from children and pets",
            "extra_tips": "Upload the full plant and a close-up of the affected area",
            "image_note": "Reliable diagnosis was not possible from this image"
        }

    if lang == "hi":
        return {
            "plant_name": "अज्ञात",
            "botanical_name": "अज्ञात",
            "confidence": "कम",
            "leaf_condition": "स्पष्ट रूप से निर्धारित नहीं हो सका",
            "possible_issue": "तस्वीर से निश्चित समस्या पहचानना संभव नहीं हुआ",
            "causes": "तस्वीर धुंधली हो सकती है या लक्षण पर्याप्त साफ नहीं हैं",
            "treatment_organic": "अच्छी रोशनी में साफ तस्वीर दोबारा अपलोड करें",
            "treatment_chemical": "केवल स्थानीय स्वीकृत दवा लेबल के अनुसार उपयोग करें",
            "use_guidance": "रासायनिक उपयोग से पहले स्थानीय विशेषज्ञ से सलाह लें",
            "safety_notes": "दवाओं को बच्चों और पालतू जानवरों से दूर रखें",
            "extra_tips": "पूरे पौधे और प्रभावित हिस्से की नज़दीकी तस्वीर दें",
            "image_note": "इस तस्वीर से भरोसेमंद पहचान ممکن نہیں ہوئی"
        }

    if lang == "ar":
        return {
            "plant_name": "غير معروف",
            "botanical_name": "غير معروف",
            "confidence": "منخفض",
            "leaf_condition": "تعذر تحديد الحالة بوضوح",
            "possible_issue": "لم يمكن تحديد مشكلة موثوقة من الصورة",
            "causes": "قد تكون الصورة غير واضحة أو الأعراض غير ظاهرة بما يكفي",
            "treatment_organic": "أعد رفع صورة أوضح وفي ضوء جيد",
            "treatment_chemical": "استخدم فقط المنتجات المعتمدة محليًا حسب الملصق",
            "use_guidance": "استشر خبيرًا محليًا قبل استخدام أي علاج كيميائي",
            "safety_notes": "أبعد المواد الكيميائية عن الأطفال والحيوانات الأليفة",
            "extra_tips": "التقط صورة للنبات كاملًا وصورة قريبة للجزء المصاب",
            "image_note": "لم يكن التشخيص الموثوق ممكنًا من هذه الصورة"
        }

    if lang == "zh":
        return {
            "plant_name": "未知",
            "botanical_name": "未知",
            "confidence": "低",
            "leaf_condition": "无法清楚判断",
            "possible_issue": "无法从图片中可靠识别问题",
            "causes": "图片可能不够清晰，或症状显示不足",
            "treatment_organic": "请在白天光线充足时重新上传更清晰的图片",
            "treatment_chemical": "仅按当地批准产品标签说明使用",
            "use_guidance": "使用化学药剂前请咨询当地专家",
            "safety_notes": "将药剂远离儿童和宠物",
            "extra_tips": "请上传整株植物和病斑特写",
            "image_note": "无法从此图片中做出可靠诊断"
        }

    # Default Urdu
    return {
        "plant_name": "نام واضح نہیں",
        "botanical_name": "نام واضح نہیں",
        "confidence": "کم",
        "leaf_condition": "واضح طور پر معلوم نہ ہو سکا",
        "possible_issue": "تصویر سے قابلِ اعتماد مسئلہ واضح نہ ہو سکا",
        "causes": "ممکن ہے تصویر واضح نہ ہو یا علامات پوری طرح نظر نہ آ رہی ہوں",
        "treatment_organic": "اچھی روشنی میں واضح تصویر دوبارہ اپلوڈ کریں اور متاثرہ حصہ قریب سے دکھائیں",
        "treatment_chemical": "صرف مقامی طور پر منظور شدہ لیبل ہدایات کے مطابق استعمال کریں",
        "use_guidance": "کیمیائی استعمال سے پہلے مقامی ماہر سے مشورہ کریں",
        "safety_notes": "ادویات بچوں اور جانوروں سے دور رکھیں",
        "extra_tips": "پورے پودے اور متاثرہ حصے کی الگ صاف تصویر دیں",
        "image_note": "اس تصویر سے قابلِ اعتماد تشخیص ممکن نہ ہو سکی"
    }


def image_to_data_uri(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_output_text(response: Any) -> str:
    # Preferred if SDK exposes output_text
    text = getattr(response, "output_text", None)
    if text and str(text).strip():
        return str(text).strip()

    # Fallback parsing
    try:
        output = getattr(response, "output", None) or []
        parts = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                txt = getattr(part, "text", None)
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts).strip()
    except Exception:
        return ""


def parse_model_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Direct JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # JSON inside ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Any first object-like block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    raise ValueError("Model output did not contain valid JSON")


def normalize_result(raw: Dict[str, Any], lang: str) -> Dict[str, str]:
    result = default_result(lang)

    expected_keys = [
        "plant_name",
        "botanical_name",
        "confidence",
        "leaf_condition",
        "possible_issue",
        "causes",
        "treatment_organic",
        "treatment_chemical",
        "use_guidance",
        "safety_notes",
        "extra_tips",
        "image_note"
    ]

    for key in expected_keys:
        if key in raw:
            result[key] = safe_str(raw.get(key), result[key])

    return result


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Plant Hospital server is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "openai_key_present": bool(OPENAI_API_KEY),
        "model": MODEL_NAME
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if not OPENAI_API_KEY:
            return jsonify({
                "error": "OPENAI_API_KEY is missing on server"
            }), 500

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        uploaded_file = request.files["image"]
        lang = normalize_lang(request.form.get("lang", "ur"))

        file_bytes = uploaded_file.read()
        if not file_bytes:
            return jsonify({
                "error": "Uploaded image is empty"
            }), 400

        mime_type = uploaded_file.mimetype or "image/jpeg"
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        data_uri = image_to_data_uri(file_bytes, mime_type)

        prompt = f"""
You are an expert plant disease analyst for farming and gardening.

Analyze the uploaded plant image carefully and return ONLY a valid JSON object.
Do not add markdown.
Do not add code fences.
Do not add explanations before or after JSON.

Language code for all user-facing values: {lang}

Rules:
1. botanical_name must remain in Latin script.
2. Keep the answer practical and cautious.
3. Never mention commercial brand names.
4. treatment_chemical should mention only active-ingredient style advice or generic chemical guidance.
5. If image is unclear, still return valid JSON with conservative wording.
6. confidence should be appropriate to the certainty level.
7. Return ALL keys below.

Required JSON shape:
{{
  "plant_name": "",
  "botanical_name": "",
  "confidence": "",
  "leaf_condition": "",
  "possible_issue": "",
  "causes": "",
  "treatment_organic": "",
  "treatment_chemical": "",
  "use_guidance": "",
  "safety_notes": "",
  "extra_tips": "",
  "image_note": ""
}}
""".strip()

        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": data_uri
                        }
                    ]
                }
            ]
        )

        model_text = extract_output_text(response)
        if not model_text:
            return jsonify({
                "error": "Model returned empty output"
            }), 500

        parsed = parse_model_json(model_text)
        final_result = normalize_result(parsed, lang)

        return jsonify(final_result), 200

    except Exception as e:
        print("ANALYZE_ERROR:", str(e))
        traceback.print_exc()

        return jsonify({
            "error": "Server analysis failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
