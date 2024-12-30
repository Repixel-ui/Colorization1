from flask import Flask, request, jsonify
from PIL import Image
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# Load the DeOldify model (PyTorch)
model = torch.hub.load('jantic/DeOldify', 'colorizer', model_dir='./models')  # Adjust path if needed

@app.route("/colorize", methods=["POST"])
def colorize():
    try:
        # Get the image from the request
        img_data = request.json['image']
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data))

        # Apply colorization    
        colorized_img = model.get_image_colorization(img)

        # Save or send back the colorized image
        img_byte_array = BytesIO()
        colorized_img.save(img_byte_array, format='JPEG')
        img_byte_array = img_byte_array.getvalue()
        img_base64 = base64.b64encode(img_byte_array).decode('utf-8')

        return jsonify({"colorized_image": img_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
