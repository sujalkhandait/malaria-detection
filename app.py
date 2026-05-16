print("Starting app...")
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import gc
from keras import backend as K
from datetime import datetime
import time
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variable initializations
model = None
MODEL_ERROR = None
MODEL_PATH = None
CLASS_NAMES = ['Parasitized', 'Uninfected']
MODEL_DISPLAY_NAME = "MobileNetV2 (Fine-Tend)"

print("\n" + "="*70)
print("🦠 MALARIA DETECTION - LOADING MODEL")
print("="*70)

# =======================
# Dynamic Model Loading (.keras)
# =======================
try:
    current_dir = os.getcwd()
    model_files = [f for f in os.listdir(current_dir) if f.endswith('.keras')]

    if model_files:
        MODEL_PATH = os.path.join(current_dir, model_files[0])
        print(f"✓ Found model: {model_files[0]}")

        # CRITICAL FIX 1: Load without compiling and do NOT call model.compile()
        # This saves hundreds of megabytes of RAM by skipping training tracking configurations
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        print(f"✓ Model Loaded Successfully (Inference Mode Only)")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
    else:
        MODEL_ERROR = "No .keras model file found"
        print(f"✗ {MODEL_ERROR}")

except Exception as e:
    MODEL_ERROR = str(e)
    print(f"✗ Error loading model: {MODEL_ERROR}")


# =======================
# Image Preprocessing
# =======================
def prepare_image(image_file):
    if model is None:
        return None

    try:
        img = Image.open(image_file)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        input_shape = model.input_shape
        img_size = (input_shape[1], input_shape[2])
        img = img.resize(img_size)

        img_array = np.array(img, dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Image processing error: {e}")
        return None


# =======================
# Routes
# =======================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model, MODEL_PATH

    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # =====================
        # Save uploaded image
        # =====================
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)

        # =====================
        # Prepare image
        # =====================
        img_array = prepare_image(image_path)

        if img_array is None:
            return jsonify({'success': False, 'error': 'Image processing failed'}), 500

        # =====================
        # Predict
        # =====================
        start_time = time.time()
        prediction = model.predict(img_array, verbose=0)
        end_time = time.time()

        prob = float(prediction[0][0])
        processing_time = round(end_time - start_time, 2)

        if prob > 0.5:
            result = "Uninfected"
            confidence = prob * 100
            parasitized_prob = (1 - prob) * 100
            uninfected_prob = prob * 100
        else:
            result = "Parasitized"
            confidence = (1 - prob) * 100
            parasitized_prob = (1 - prob) * 100
            uninfected_prob = prob * 100

        model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "Loaded Model"
        model_name = model_name.replace('.keras', '')

        # =======================================
        # CRITICAL FIX 2: FORCE SYSTEM MEMORY PURGE
        # =======================================
        del img_array
        K.clear_session()
        gc.collect()

        # =======================================
        # Return Structured template data response
        # =======================================
        return render_template(
            'result.html',
            prediction=result,
            confidence=round(confidence, 2),
            parasitized_prob=round(parasitized_prob, 2),
            uninfected_prob=round(uninfected_prob, 2),
            processing_time=processing_time,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            image_file=file.filename,
            model_name=model_name
        )

    except Exception as e:
        # Run cleanup routines even if code errors out mid-execution
        K.clear_session()
        gc.collect()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })


# =======================
# Universal Deployment Entry-Point
# =======================
if __name__ == '__main__':
    print("\n" + "="*70)
    print(f"{'✓' if model else '✗'} Model Status Check: {'LOADED' if model else 'NOT LOADED'}")
    print("="*70 + "\n")

    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)