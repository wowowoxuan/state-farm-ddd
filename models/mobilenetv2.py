import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, n_classes, use_pretrained):
        super().__init__()
        self.base_model = models.mobilenet_v2(pretrained = use_pretrained).features  # take the model without classifier  # size of the layer before classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=62720, out_features=12000),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=12000, out_features=2000),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=2000, out_features=n_classes),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=62720, out_features=n_classes),

        )

    def forward(self, x):
        x = self.base_model(x)
        # x = self.pool(x)
        
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        # print(x.shape)
        output = self.classifier(x)
        return output

class ResNet18(nn.Module):
    def __init__(self, n_classes, use_pretrained):
        super().__init__()
        # print(models.resnet18())

        self.base_model = models.resnet18(pretrained = use_pretrained)  # take the model without classifier  # size of the layer before classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=62720, out_features=12000),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=12000, out_features=2000),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=2000, out_features=n_classes),
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1000, out_features=n_classes),

        )

    def forward(self, x):
        x = self.base_model(x)
        # x = self.pool(x)
        
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        # print(x.shape)
        output = self.classifier(x)
        return output

class VGG16(nn.Module):
    def __init__(self, n_classes, use_pretrained):
        super().__init__()
        self.base_model = models.vgg16(pretrained = use_pretrained).features  # take the model without classifier  # size of the layer before classifier
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(

            nn.Linear(in_features=25088, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_classes),

        )
        # self.base_model.classifier 

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)
        
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        # print(x.shape)
        output = self.classifier(x)
        return output

class AlexNet(nn.Module):
    def __init__(self, n_classes, use_pretrained):
        super().__init__()
        # print(models.alexnet())
        self.base_model = models.alexnet(pretrained=use_pretrained)#.features  # take the model without classifier  # size of the layer before classifier
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1000, out_features=n_classes),#9216
        )

    def forward(self, x):
        x = self.base_model(x)
        # x = self.pool(x)
        
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        # print(x.shape)
        output = self.classifier(x)
        return output

class tinyNet(nn.Module):
    def __init__(self):
        super(tinyNet,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(16,32,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(32,64,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(64,128,3,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=28800, out_features=9)
        )
    def forward(self, x):
        x = self.conv_block(x)
        # x = self.pool(x)
        
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        # print(x.shape)
        output = self.classifier(x)
        return output

if __name__ == '__main__':
    model = ResNet18(n_classes=10,use_pretrained=True)