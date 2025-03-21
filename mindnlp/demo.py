from mindnlp.transformers import GLPNImageProcessor, GLPNForDepthEstimation
import mindspore as ms
import numpy as np
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-kitti")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

# prepare image for the model
inputs = processor(images=image, return_tensors="ms")


outputs = model(**inputs)
predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = ms.ops.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth.show()
