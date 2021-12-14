#Image Encoder
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    

    def __init__(self, output_size):
        
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, images):
        
        features = self.cnn(images)
        output = self.bn(features)
        return output
