from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import json

model = None
interpreter = None
input_index = None
output_index = None

class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]

BUCKET_NAME = "thanishk-model-901417"


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model

    # CORS Headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    }

    # Handle preflight requests
    if request.method == "OPTIONS":
        return ("", 204, headers)

    print("Function started")
    print("Content-Type:", request.content_type)
    print("Files:", request.files)
    print("Files keys:", list(request.files.keys()))
    print("Form:", request.form)
    print("Form keys:", list(request.form.keys()))

    try:

        if model is None:
            print("Downloading model")

            download_blob(
                BUCKET_NAME,
                "models/tomato_leaf_disease_model.h5",
                "/tmp/tomato_leaf_disease_model.h5",
            )

            print("Loading model")

            model = tf.keras.models.load_model(
                "/tmp/tomato_leaf_disease_model.h5"
            )

            print("Model loaded successfully")

        if "data" not in request.files:
            return (
                json.dumps({
                    "error": "No file uploaded",
                    "received_files": list(request.files.keys()),
                    "content_type": request.content_type
                }),
                400,
                headers
            )

        image = request.files["data"]

        print("Image received")

        image = np.array(
            Image.open(image)
            .convert("RGB")
            .resize((128, 128))
        )

        img_array = tf.expand_dims(image, 0)

        predictions = model.predict(img_array)

        print("Predictions:", predictions)
        print("Predictions:", predictions.tolist())

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(
            100 * float(np.max(predictions[0])),
            2
        )

        return (
            json.dumps({
                "class": predicted_class,
                "confidence": confidence
            }),
            200,
            headers
        )

    except Exception as e:

        print("ERROR:", str(e))

        return (
            json.dumps({
                "error": str(e)
            }),
            500,
            headers
        )