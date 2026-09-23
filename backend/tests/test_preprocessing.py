from io import BytesIO
from PIL import Image
from app.services.preprocessing import preprocess_image

def test_preprocessing_shape_and_range():
    image = Image.new("RGB", (20, 30), "white"); output = preprocess_image(BytesIO(_png(image)))
    assert output.shape == (224, 224, 1); assert output.dtype.name == "float32"; assert 0 <= output.min() <= output.max() <= 1

def _png(image):
    buffer = BytesIO(); image.save(buffer, format="PNG"); return buffer.getvalue()
