from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "Plant Hospital server is running"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        lang = request.form.get("lang", "ur")

        prompt = f"""
You are a plant disease expert.
Analyze the uploaded plant image and reply only in valid JSON.

Language code: {lang}

Return JSON with these keys:
plant_name
botanical_name
confidence
leaf_condition
possible_issue
causes
treatment_organic
treatment_chemical
use_guidance
safety_notes
extra_tips
image_hint
"""

        uploaded = client.files.create(
            file=(image.filename, image.stream, image.mimetype),
            purpose="vision"
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "file_id": uploaded.id
                        }
                    ]
                }
            ]
        )

        try:
            ai_text = response.output[0].content[0].text
        except:
            ai_text = response.output_text

        return app.response_class(
            response=ai_text,
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
