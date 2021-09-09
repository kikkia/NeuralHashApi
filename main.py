import onnxruntime
import numpy as np
from PIL import Image
from flask import Flask, request
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Load Onnx
onnxSession = onnxruntime.InferenceSession("resources/model.onnx")

# Load output hash matrix
seed1 = open("resources/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


@app.route('/hash/link/', methods=['POST'])
async def link_hash():
    json = request.get_json(force=True)
    if not json['url']:
        return 'No image url specified', 400
    response = requests.get(json['url'])
    img = Image.open(BytesIO(response.content))
    # Preprocess image
    return get_hash(img.convert('RGB'))


@app.route('/hash/upload/', methods=['POST'])
def upload_hash():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            return 'No file in request', 400
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No filename', 400
        if allowed_file(file.filename):
            return get_hash(Image.open(file).convert('RGB'))
        else:
            return 'File format not allowed. Allowed: ' + str(ALLOWED_EXTENSIONS), 400

def get_hash(image):
    resized = image.resize([360, 360])
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])

    # Run model
    inputs = {onnxSession.get_inputs()[0].name: arr}
    outs = onnxSession.run(None, inputs)

    # Convert model output to hex hash
    hash_output = seed1.dot(outs[0].flatten())
    hash_bits = ''.join(['1' if it >= 0 else '0' for it in hash_output])
    hash_hex = '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)

    return hash_hex

def allowed_file(filename):
    return os.path.splitext(filename)[1] in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(port=os.getenv("SERVER_PORT", 80))
