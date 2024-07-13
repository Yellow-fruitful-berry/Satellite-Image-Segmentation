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
    # Convert image to the required format for the model
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# def postprocess_output(output):
#     # Convert model output to an image format
#     output = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
#     output = np.argmax(output, axis=0)  # Assuming you have multiple classes
#     output_image = Image.fromarray(output.astype(np.uint8))
#     return output_image


file = open('100877_sat.jpg', 'rb')
image = Image.open(file)

input_tensor = preprocess_image(image).to(DEVICE)
with torch.no_grad():
    output = model(input_tensor)

output_image = output[0]# .permute(1, 2, 0)

class_rgb_values = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]

image = preprocess_image(image)
image = image.to('cpu')

segmodel = model
# Creating a batch of size 1 (unsqueeze(0)) for image
pred = segmodel.forward(image)

# Getting pred tensor from gpu to cpu, squeezing it and casting it to numpy
pred = pred.cpu().detach().squeeze().numpy()

# Getting gt_mask tensor from gpu to cpu and casting to uint8

# reversing one_hot transformations (made in the Mining class object) of gt_mask and pred
reversed_pred = reverse_one_hot(pred)

# reapplying the original colors in the reversed one hot images
# and getting them as uint8
prediction = colour_code_segmentation(reversed_pred, class_rgb_values).astype(np.uint8)

# visualize(
#     # image=reverse_transform (imagevis),
#     prediction=reverse_transform(prediction),
#     # one_hot_mask=reverse_transform(reversed_gt_mask.astype(np.uint8)),
# )


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify(error='No file uploaded'), 400

    # Test with a local file
    # file = request.files['file']
    file = open('100877_sat.jpg', 'rb')
    image = Image.open(file)

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
