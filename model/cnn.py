import torch.nn as nn
from torch.utils import data
from dataset import VoiceDataset
from tqdm import tqdm

CATEGORY_NUM = 7
TAG_NUM = 15

class CategoryCNN(nn.Module):
    def __init__(self):
        super(CategoryCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=5),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=20),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(20*14*47, 764),
            nn.ReLU(),
            nn.Linear(764, 64),
            nn.ReLU(),
            nn.Linear(64, CATEGORY_NUM)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 20*14*47)
        x = self.softmax(self.fc(x))
        return x



class TagCNN(nn.Module):
    def __init__(self):
        super(TagCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=5),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=20),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(20*14*47, 764),
            nn.ReLU(),
            nn.Linear(764, 64),
            nn.ReLU(),
            nn.Linear(64, TAG_NUM)
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 20*14*47)
        x = self.softmax(self.fc(x))
        return x


def test():
    voiceDataset = VoiceDataset(sampleNumber=100)
    voiceDataGenerator = data.DataLoader(voiceDataset, batch_size=8, shuffle=False)
    model = TagCNN()
    for batch in tqdm(voiceDataGenerator):
        melspec, category, tag = batch
        output = model(melspec.unsqueeze(1))

if __name__ == '__main__':
    test()