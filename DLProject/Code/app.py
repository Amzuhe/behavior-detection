import base64
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
import tensorflow as tf
from tensorflow import keras


IMG_SIZE = (224, 224)
MODEL_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [
    "distracted",
    "fatigue",
    "focused",
    "raise_hand",
    "sleeping",
    "using_smartphone",
    "writing_reading",
]
NUM_CLASSES = len(CLASS_NAMES)
NORMAL_CLASSES = {"focused", "writing_reading"}
SUSPICIOUS_CLASSES = {"distracted", "fatigue", "sleeping", "using_smartphone"}


def decode_base64_image(payload: str) -> bytes:
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def build_inference_model() -> keras.Model:
    data_augmentation = keras.Sequential(
        [keras.layers.RandomFlip("horizontal")], name="data_augmentation"
    )
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(2e-4),
    )(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)


app = Flask(__name__)

if os.getenv("PROCTOR_ENABLE_GPU", "0") != "1":
    tf.config.set_visible_devices([], "GPU")


def resolve_model_path(model_dir: Path) -> Path:
    preferred_names = [
        "final_proctor_model.keras",
        "final_proctor_model.h5",
        "best_final_proctor_model_finetune.h5",
    ]
    candidates = [model_dir / name for name in preferred_names if (model_dir / name).exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No supported model file found in {model_dir}. "
            "Expected one of: final_proctor_model.keras, final_proctor_model.h5, "
            "best_final_proctor_model_finetune.h5"
        )

    candidates.sort(
        key=lambda path: (
            0 if path.name.startswith("final_proctor_model") else 1,
            -path.stat().st_mtime,
        )
    )
    selected = candidates[0]
    print(f"Using model file: {selected.name} (modified {selected.stat().st_mtime})")
    return selected


def load_inference_model(model_path: Path) -> keras.Model:
    try:
        loaded = keras.models.load_model(model_path, compile=False)
        print(f"Loaded full model from: {model_path}")
        return loaded
    except Exception as full_model_error:
        print(f"Full-model load failed, trying weights path: {full_model_error}")

    fallback = build_inference_model()
    fallback.load_weights(model_path)
    print(f"Loaded weights into inference architecture from: {model_path}")
    return fallback

MODEL_PATH = resolve_model_path(MODEL_DIR)
model = load_inference_model(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    if not image_b64:
        return jsonify({"error": "Missing 'image' field in request body."}), 400

    try:
        image = decode_base64_image(image_b64)
        input_tensor = preprocess_image(image)
        probs = model.predict(input_tensor, verbose=0)[0]
    except Exception as exc:
        return jsonify({"error": f"Failed to process image: {exc}"}), 400

    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else pred_idx
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    second_confidence = float(probs[second_idx])
    margin = confidence - second_confidence
    if pred_class in SUSPICIOUS_CLASSES:
        is_suspicious = True
    else:
        is_suspicious = False

    return jsonify(
        {
            "predicted_class": pred_class,
            "confidence": confidence,
            "margin": margin,
            "second_predicted_class": CLASS_NAMES[second_idx],
            "second_confidence": second_confidence,
            "scores": [
                {
                    "label": class_name,
                    "confidence": float(prob),
                }
                for class_name, prob in zip(CLASS_NAMES, probs)
            ],
            "is_suspicious": is_suspicious,
            "suspicious_type": pred_class if is_suspicious else None,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
