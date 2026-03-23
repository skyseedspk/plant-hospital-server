from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Plant Hospital API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded. Expected form-data field name: file"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        # ابھی dummy result دے رہا ہوں تاکہ app side test ہو جائے
        # بعد میں یہاں اصل AI model لگے گا
        return jsonify({
            "success": True,
            "plant": "Detected plant image",
            "disease": "Analysis placeholder",
            "confidence": "90%",
            "treatment": "Use proper diagnosis logic or AI model here.",
            "warning": "Consult your local agriculture expert before applying any chemical."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
