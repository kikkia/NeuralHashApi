import onnxruntime
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, APIRouter, Request
from fastapi.templating import Jinja2Templates
from ddtrace.contrib.asgi import TraceMiddleware
from pydantic import BaseModel
from pathlib import Path
from content_size_limit_asgi import ContentSizeLimitMiddleware
from content_size_limit_asgi.errors import ContentSizeExceeded
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
import hashlib

app = FastAPI()

# Load Onnx
onnxSession = onnxruntime.InferenceSession("app/resources/model.onnx")

# Load output hash matrix
seed1 = open("app/resources/neuralhash_128x96_seed1.dat", 'rb').read()[128:]
seed1 = np.frombuffer(seed1, dtype=np.float32)
seed1 = seed1.reshape([96, 128])

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))
SERVICE_NAME = "neuralhash-api"
DD_ENABLED = False
general_pages_router = APIRouter()


class Hash_Url_Request(BaseModel):
    url: str


@app.get("/", status_code=200)
async def home(request: Request):
    return TEMPLATES.TemplateResponse("homepage.html", {"request": request})


@app.post('/api/link')
async def link_hash(request: Hash_Url_Request):
    if not request.url:
        raise HTTPException(status_code=400, detail='No image url specified')
    if not allowed_url(request.url):
        raise HTTPException(status_code=400, detail="Resource at url is not a supported format (Supported: " +
                                                    str(ALLOWED_EXTENSIONS) + ")")
    response = requests.get(request.url)
    # Preprocess image
    try:
        img = Image.open(BytesIO(response.content))
        hash = get_hash(img.convert('RGB'))
        md5 = getMD5(img.convert('RGB'))
        return {"hash": hash, "md5": md5}
    except UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail="Linked image file is corrupt")


@app.post('/api/upload')
async def upload_hash(file: UploadFile = File(...)):
    # check if the post request has the file part
    if not file:
        raise HTTPException(status_code=400, detail='No file provided')
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        raise HTTPException(status_code=400, detail='No filename')
    if allowed_file(file.filename):
        try:
            img = Image.open(file.file).convert('RGB')
            hash = get_hash(img)
            md5 = getMD5(img.convert('RGB'))
            return {"hash": hash, "md5": md5}
        except UnidentifiedImageError as e:
            raise HTTPException(status_code=400, detail="Corrupt image file")

    else:
        raise HTTPException(status_code=400, detail="Provided file is not a supported format (Supported: " +
                                                    str(ALLOWED_EXTENSIONS) + ")")


@app.exception_handler(ContentSizeExceeded)
async def unicorn_exception_handler(request: Request, exc: ContentSizeExceeded):
    raise HTTPException(status_code=400, detail="Provided file is too large. Max size 50MB")


@app.exception_handler(StarletteHTTPException)
async def exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return TEMPLATES.TemplateResponse('404.html', {'request': request})
    else:
        # Just use FastAPI's built-in handler for other errors
        return await http_exception_handler(request, exc)


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

def getMD5(image):
    return hashlib.md5(image.tobytes()).hexdigest()


def allowed_file(filename):
    return os.path.splitext(filename)[1].replace('.', '') in ALLOWED_EXTENSIONS


def allowed_url(path):
    for allowedFormat in ALLOWED_EXTENSIONS:
        if str(path).endswith(allowedFormat):
            return True

    return False


app.add_middleware(ContentSizeLimitMiddleware, max_content_size=51200000)

# Add middlewares for asgi
if os.environ.get("DD_ENABLE", DD_ENABLED):
    app = TraceMiddleware(app)

if __name__ == '__main__':
    port = os.getenv("SERVER_PORT", 80)
    print('starting server on port: ' + str(port))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=4)
