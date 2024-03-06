import torch
import torch.jit
import numpy as np
from torch import nn
from PIL import Image
import torch.optim as optim
from torchvision import transforms

from torchvision.transforms.functional import get_image_size

data_mean = [0.48464295, 0.4219926, 0.39299254]
data_std = [0.32095764, 0.30016821, 0.29993387]

det_resized = (224, 224)    # The normalized size of the images in the detection model
rec_resized = det_resized   # The normalized size of the images in the recognition model
prediction_threshold = 0.0

det_tr_transform = transforms.Compose([
    # transforms.Resize(300, 300),
    transforms.RandomResizedCrop(det_resized),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(20),
    transforms.ToTensor(),
    # transforms.Normalize(mean=train_mean, std=train_std)
    transforms.Normalize(mean=data_mean, std=data_std)
])

det_base_transform = transforms.Compose([
    transforms.Resize(det_resized),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])

rec_tr_transform = transforms.Compose([
    transforms.RandomResizedCrop(rec_resized),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomRotation(180),
    transforms.RandomGrayscale(0.2),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    transforms.RandomAffine(90),
    transforms.AutoAugment(),
    ## scaling shearing affine
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])

rec_val_transform = transforms.Compose([
    transforms.Resize(rec_resized),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])

detection_cnn_layers = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
detection_fc_layers = nn.Sequential(
    # nn.Dropout(0.3),
    nn.Linear(25088, 32),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(32, 4)
)

recognition_cnn_layers = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2)
)
recognition_fc_layers = nn.Sequential(
    # nn.Dropout(0.2),# prueba de quitar dropout
    nn.Linear(21632, 30), # image size que no sea pequeño >12x12
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(30, 81), # 1-80 are ids + (-1) are 81 identities
)

class CNN(nn.Module):
    def __init__(self, cnn_layers, fc_layers, type):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers
        self.cnn_type = type

    def forward(self, data):
        output = self.cnn_layers(data)
        output = self.flatten(output) # before linear layer !!!
        output = self.fc_layers(output)
        return output

    def predict(self, test_image):
        self.eval()
        with torch.inference_mode(mode=True):
            test_image = test_image.convert('RGB')
            w, h = get_image_size(test_image)
            if self.cnn_type == 'detection':
                test_image = det_base_transform(test_image)
                # test_image = torch.tensor(test_image, dtype=torch.float32)
                output = self.forward(test_image.unsqueeze(0))[0]
                # print(output)
                # output = [output[0], output[1]*w/det_resized[0], output[2]*h/det_resized[1], output[3]*w/det_resized[0], output[4]*h/det_resized[1]]
                output = [output[0]*w/det_resized[0], output[1]*h/det_resized[1], output[2]*w/det_resized[0], output[3]*h/det_resized[1]]
                # print(output)
                return output
            elif self.cnn_type == 'recognition':
                test_image = rec_val_transform(test_image)
                output = self.forward(test_image.unsqueeze(0))
                output = torch.softmax(output, 1)
                prediction = np.argmax(output)
                if prediction == 0 or output[0][prediction] < prediction_threshold:
                    return -1
                return int(prediction)
            
def myCrop(image: Image, bounds_tensors):
    w, h = image.size
    bounds = [t.to(torch.int32).item() for t in bounds_tensors]

    for i in range(0, len(bounds), 2):
        if bounds[i] < 0:
            bounds[i] = 0
        elif bounds[i] > w:
            bounds[i] = w

    for i in range(1, len(bounds)+1, 2):
        if bounds[i] < 0:
            bounds[i] = 0
        elif bounds[i] > h:
            bounds[i] = h

    image = image.crop(bounds)
    return image