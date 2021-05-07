import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
import torch.nn as nn

from dataset import TestDataset, MaskBaseDataset

class Resnet50(nn.Module):
  def __init__(self, num_classes=18, pretrained=False):
    super(Resnet50, self).__init__()
    self.model = resnet50(pretrained)
    n_features = self.model.fc.in_features
    self.model.fc = nn.Linear(n_features, num_classes)
  
  def forward(self, x):
    x = self.model(x)
    return x

def load_model_EfficientNet(model_name, model_path, num_classes, device):
    
    model = EfficientNet.from_name(model_name, num_classes=18)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_model_resnet50(model_path, num_classes, device):
    
    model = Resnet50(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model


def load_model_Vgg16(model_path, num_classes, device):
  
    model = Vgg16(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, output_dir, args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model_A = load_model_EfficientNet('efficientnet-b3', '/opt/ml/model/efficientb3_processed0_train_epochs10_aug/4.pth', num_classes, device).to(device)
    model_B = load_model_EfficientNet('efficientnet-b0', '/opt/ml/model/efficientb0_processed1_train_epochs10_aug/9.pth', num_classes, device).to(device)
    model_C = load_model_resnet50('/opt/ml/model/resnet_processed2_train_epochs10_aug/best.pth', num_classes, device).to(device)
    model_D = load_model_EfficientNet('efficientnet-b0', '/opt/ml/model/efficientb0_processed3_train_epochs10_aug/9.pth', num_classes, device).to(device)
    model_E = load_model_EfficientNet('efficientnet-b0', '/opt/ml/model/efficientb0_processed4_train_epochs10_aug/best.pth', num_classes, device).to(device)
    model_A.eval()
    model_B.eval()
    model_C.eval()
    model_D.eval()
    model_E.eval()


    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred_A = model_A(images)
            pred_B = model_B(images)
            pred_C = model_C(images)
            pred_D = model_D(images)
            pred_E = model_E(images)
            pred = pred_A + pred_B + pred_C + pred_D + pred_E
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, output_dir, args)
