import torch
import torch.nn as nn
import timm
from config import Config

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = (in_channels // 8) ** -0.5

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key) * self.scale
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out + x  # skip connection

class CustomTNT(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = Config.OUTPUT_SIZE
        self.backbone = timm.models.tnt_s_patch16_224(pretrained=False)
        state_dict = torch.load(Config.MODEL_SAVE_PATH)
        self.backbone.load_state_dict(state_dict)
        self.backbone.reset_classifier(0)
        out_channels = 384  # TNT-S model embedding dimension

        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            SelfAttention(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool2d(self.output_size)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        batch_size, num_patches, embed_dim = x.size()

        # Remove the classification token
        x = x[:, 1:, :]

        num_patches -= 1
        height = width = int(num_patches**0.5)
        assert height * width == num_patches

        x = x.permute(0, 2, 1).contiguous().view(batch_size, embed_dim, height, width)
        x = self.up(x)
        x = torch.squeeze(x, dim=1)
        return x