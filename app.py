from flask import Flask, request, jsonify, send_file, send_from_directory
import torch
from PIL import Image
import io
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import multiprocessing as mp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import os
import torchvision.transforms as transforms
from torch import nn
from matplotlib import pyplot as plt
from io import BytesIO
from flask import Flask, render_template, send_file, request, jsonify, send_from_directory
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig, bbox_to_dimensions
from io import BytesIO
import os
import numpy as np
import torch
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import multiprocessing as mp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
# from torch import nn
from matplotlib import pyplot as plt
from io import BytesIO

import torch.nn as nn
from torchvision.transforms import transforms

IMG_SIZE = (512, 512)

app = Flask(__name__)
ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CHANNELS = 3
CLASSES = 7
ACTIVATION = 'sigmoid'
LR = 0.001
BATCH_SIZE = 8
DEVICE, NUM_DEVICES = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
WORKERS = mp.cpu_count()

net = smp.UnetPlusPlus(
    encoder_name=ENCODER,                # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,                # set to None to avoid downloading
    in_channels=CHANNELS,                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=CLASSES,                     # model output channels (number of classes in your dataset)
    activation=ACTIVATION
)

# Manually load the encoder weights
weights_path = './efficientnet-b3-5fb5a3c3.pth'
encoder_weights = torch.load(weights_path)
net.encoder.load_state_dict(encoder_weights)

preprocess_input = get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)

LOSS = nn.CrossEntropyLoss()


class SegmentationModel(pl.LightningModule):
    def __init__(self, net, loss, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss = loss
        self.net = net

    def forward(self, x):
        return self.net(x)


# Load the model
model = SegmentationModel(net=net, loss=LOSS, lr=LR)
model.load_state_dict(torch.load('/Users/roman/Desktop/StemGNN/classification_website/lightning_trained-v1.ckpt', map_location=torch.device('cpu')), strict=False)
model.eval()

# Configure Sentinel Hub
config = SHConfig()
config.sh_client_id = '72384d45-3b1a-4b1a-9915-7f69ae01e125'
config.sh_client_secret = '10oJej5guXjV1888E5UJmy6uLaAHBeaH'


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal (label, colour)
        class_map = np.all (equality, axis=-1)
        semantic_map.append (class_map)
    semantic_map = np.stack (semantic_map, axis=-1)

    return semantic_map


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax (image, axis=0)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array (label_values)
    x = colour_codes[image.astype (int)]

    return x


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


reverse_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=IMG_SIZE),
        ])


def preprocess_image(image):
    if isinstance (image, np.ndarray):
        image = Image.fromarray (image)
    # Convert image to the required format for the model
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)

    # If the image has 4 channels (RGBA), convert it to RGB
    if image_tensor.shape[0] == 4:
        image_tensor = image_tensor[:3, :, :]  # Keep only RGB channels

    # Normalize the image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = normalize(image_tensor)

    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def get_image(bbox_minx, bbox_miny, bbox_maxx, bbox_maxy, time_start, time_end):
    time_range = (time_start, time_end)

    evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
            """

    betsiboka_coords_wgs84 = (bbox_minx, bbox_miny, bbox_maxx, bbox_maxy)

    resolution = 60
    betsiboka_bbox = BBox (bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions (betsiboka_bbox, resolution=resolution)
    size = (512, 512)
    request = SentinelHubRequest (
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data (
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(time_start, time_end),
            )
        ],
        responses=[SentinelHubRequest.output_response ("default", MimeType.PNG)],
        bbox=betsiboka_bbox,
        size=size,
        config=config,
    )

    response = request.get_data()

    # print(len(response))
    # print(response[0].shape)

    image = response[0]
    # print(type(image))

    return image


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/segment', methods=['POST'])
def segment():
    bbox_minx = float (request.form['bbox_minx'])
    bbox_miny = float (request.form['bbox_miny'])
    bbox_maxx = float (request.form['bbox_maxx'])
    bbox_maxy = float (request.form['bbox_maxy'])
    time_start = request.form['time_start']
    time_end = request.form['time_end']

    image = get_image(bbox_minx, bbox_miny, bbox_maxx, bbox_maxy, time_start, time_end)

    input_tensor = preprocess_image (image).to (DEVICE)
    with torch.no_grad ():
        output = model (input_tensor)

    output_image = output[0]  # .permute(1, 2, 0)

    class_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255],
                        [0, 0, 0]]

    image = preprocess_image (image)
    image = image.to ('cpu')

    segmodel = model
    # Creating a batch of size 1 (unsqueeze(0)) for image
    pred = segmodel.forward (image)

    # Getting pred tensor from gpu to cpu, squeezing it and casting it to numpy
    pred = pred.cpu().detach().squeeze().numpy()

    # Getting gt_mask tensor from gpu to cpu and casting to uint8

    # reversing one_hot transformations (made in the Mining class object) of gt_mask and pred
    reversed_pred = reverse_one_hot(pred)

    # reapplying the original colors in the reversed one hot images
    # and getting them as uint8
    prediction = colour_code_segmentation(reversed_pred, class_rgb_values).astype (np.uint8)

    prediction = reverse_transform (prediction)

    # Save the PIL image to a BytesIO object
    byte_io = BytesIO ()
    prediction.save (byte_io, 'PNG')
    byte_io.seek (0)

    return send_file (byte_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
