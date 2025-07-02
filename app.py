from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
from utils.predict import load_all_models, predict_image  # Updated import name

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model at startup
model_info = load_all_models()  # Changed to match function name

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_info=model_info)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction=None, error="No file uploaded", model_info=model_info)

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction=None, error="No file selected", model_info=model_info)

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        prediction = predict_image(filepath, model_info)
    except Exception as e:
        return render_template("index.html", prediction=None, error=f"Prediction error: {e}", model_info=model_info)

    image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template("index.html", 
                         prediction=prediction, 
                         image_url=image_url,
                         model_info=model_info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)