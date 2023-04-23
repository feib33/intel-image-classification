#!/usr/bin/env python3
import timm
from torchvision import datasets
from data_transform import data_transforms
import util_logger as L
import argparse
import warnings
import torch
import os

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Intel Image Classification')
    parser.add_argument("--bsize", type=int, help="batch size", default=32)
    parser.add_argument("--pretrained", type=bool, help="train or test", default=False)
    parser.add_argument("--model_name", type=str, help="model name", default='MConvMixer')
    args = parser.parse_args()
    return args


def pred_model():
    args = parse_args()
    data_dir = './data/archive'
    result_dir = './results'
    model_dir = './model_zoo'
    pred_log_file = args.model_name + '_pred.log'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'pred'), data_transforms['test'])
    L.logger_info(pred_log_file, log_path=os.path.join(result_dir, pred_log_file))

    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=args.bsize,
                                             shuffle=False, num_workers=4)

    if args.model_name == 'MConvMixer':
        from networks import MConvMixer
        model = MConvMixer.MConvMixer(dims=(384, 768, 1536), depths=(6, 10, 6),
                                      patch_size=(7, 7), n_class=6)

    else:
        model = timm.create_model(args.model_name, pretrained=False,
                                  num_classes=6, input_size=(3,224,224))

    # if use pretrained model, the suffix becomes 'best' other wise 'bestn'
    suffix = 'best' if args.pretrained else 'bestn'
    state_dict = torch.load(os.path.join(model_dir, '{}_{}.pth'.format(args.model_name, suffix)))
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    print("total parameters: {:.2f}M".format(total_params / 1e6))
    model.to(device)
    model.eval()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use device: {}".format(device))
    pred_model()
