import onnxruntime
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Load Onnx
onnxSession = onnxruntime.InferenceSession("app/resources/model.onnx")

# Load output hash matrix
seed1 = open("app/resources/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


class hashUrlRequest(BaseModel):
    url: str


@app.post('/hash/link/')
async def link_hash(request: hashUrlRequest):
    if not request.url:
        raise HTTPException(status_code=400, detail='No image url specified')
    if not allowed_url(request.url):
        raise HTTPException(status_code=400, detail="Resource at url is not a supported format (Supported: " +
                                                    str(ALLOWED_EXTENSIONS) + ")")
    response = requests.get(request.url)
    img = Image.open(BytesIO(response.content))
    # Preprocess image
    return {"hash": get_hash(img.convert('RGB'))}


@app.post('/hash/upload/')
async def upload_hash(file: UploadFile = File(...)):
    # check if the post request has the file part
    if not file:
        raise HTTPException(status_code=400, detail='No file provided')
    print(file.filename)
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        raise HTTPException(status_code=400, detail='No filename')
    if allowed_file(file.filename):
        return {"hash": get_hash(Image.open(file.file).convert('RGB'))}
    else:
        raise HTTPException(status_code=400, detail="Provided file is not a supported format (Supported: " +
                                                    str(ALLOWED_EXTENSIONS) + ")")

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
    return os.path.splitext(filename)[1].replace('.', '') in ALLOWED_EXTENSIONS

def allowed_url(path):
    for allowedFormat in ALLOWED_EXTENSIONS:
        if str(path).endswith(allowedFormat):
            return True

    return False


if __name__ == '__main__':
    port = os.getenv("SERVER_PORT", 80)
    print('starting server on port: ' + str(port))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=4)
