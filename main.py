from collections import OrderedDict
from torch import nn
import torch
import os
import torchvision
from robustbench.utils import download_gdrive
from adv_lib.attacks.fast_minimum_norm_mod import fmn


class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()
        self.num_channels = 1
        self.num_labels = 10
        activ = nn.ReLU(True)
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


# download model
PRETRAINED_FOLDER = 'pretrained'
# create folder for storing models
if not os.path.exists(PRETRAINED_FOLDER):
    os.mkdir(PRETRAINED_FOLDER)
MODEL_ID_REGULAR = '12HLUrWgMPF_ApVSsWO4_UHsG9sxdb1VJ'
filepath = os.path.join(PRETRAINED_FOLDER, f'mnist_regular.pth')
if not os.path.exists(filepath):
    # utility function to handle google drive data
    download_gdrive(MODEL_ID_REGULAR, filepath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN()
model.load_state_dict(torch.load(os.path.join(PRETRAINED_FOLDER,
                                              'mnist_regular.pth'), map_location=device))
model.eval()

BATCH_SIZE = 10
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)


#-----------------------------------------------------------------------------------------#


def accuracy(model, samples, labels):
    preds = model(samples)
    acc = (preds.argmax(dim=1) == labels).float().mean()
    return acc.item()


samples, labels = next(iter(dl_test))

acc = accuracy(model, samples, labels)

print("standard accuracy: ", acc)

x_adv = samples.clone()

advs = fmn(model, x_adv, labels, norm=0)
acc = accuracy(model, advs, labels)
print("Robust accuracy", acc)

