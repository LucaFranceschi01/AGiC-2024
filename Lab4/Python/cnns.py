import torch
import torch.jit
import numpy as np
from torch import nn
from PIL import Image
import torch.optim as optim
from torchvision import transforms

from torchvision.transforms.functional import get_image_size

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

det_resized = (224, 224)    # The normalized size of the images in the detection model
rec_resized = det_resized   # The normalized size of the images in the recognition model

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
    transforms.Resize(rec_resized),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
])

rec_val_transform = rec_tr_transform

detection_cnn_layers = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
detection_fc_layers = nn.Sequential(
    nn.Linear(25088, 16),
    nn.ReLU(inplace=True),
    nn.Linear(16, 4)
)

recognition_cnn_layers = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
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
    nn.MaxPool2d(kernel_size=2, stride=2)
)
recognition_fc_layers = nn.Sequential(
    nn.Linear(12544, 50),
    nn.ReLU(inplace=True),
    nn.Linear(50, 81), # 1-80 are ids + (-1) are 81 identities
    nn.Softmax(0)
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
                output = np.argmax(output)
                if output == 0:
                    return -1
                return int(output)
            
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