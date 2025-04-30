import os
import csv
import argparse
import torch
import joblib
import numpy as np
import rasterio
from glob import glob
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from UNet import UNet
import rle_encoder_decoder as rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['unet', 'rf'], required=True)
    args = parser.parse_args()

    TEST_DIR = 'test'
    OUT_CSV = '3.csv'
    LOG_FILE = 'model_logs.txt'
    UNET_WEIGHT_PATH = 'models_weights/unet.pkl'
    RF_WEIGHT_PATH = 'models_weights/random_forest.pkl'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = csv.writer(open(OUT_CSV, 'w', newline=''))
    writer.writerow(['id', 'segmentation'])

    if args.model == 'unet':
        model = UNet(in_channels=4, out_channels=1).to(DEVICE)
        state = torch.load(UNET_WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof_op:
            dummy = torch.randn(1, 4, 256, 256).to(DEVICE)
            _ = model(dummy)
            prof_op.step()
        ops = sum(e.count for e in prof_op.key_averages())

        with open(LOG_FILE, 'w') as f:
            f.write(f"UNet Parameters: {num_params}\nUNet Operations: {ops}\n")

        for path in glob(os.path.join(TEST_DIR, '*.tif')):
            with rasterio.open(path) as src:
                img = src.read().astype(np.float32) / 255.0
                id_ = os.path.splitext(os.path.basename(path))[0]
            inp = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
            out = model(inp)
            mask = (out > 0.5).cpu().numpy().squeeze().astype(np.uint8)
            writer.writerow([id_, rle.rle_encode(mask)])

    else:
        rf = joblib.load(RF_WEIGHT_PATH)
        # Count of Params & OPS
        num_params = sum(t.tree_.node_count for t in rf.estimators_)
        ops = num_params

        with open(LOG_FILE, 'w') as f:
            f.write(f"RandomForest Parameters (nodes): {num_params}\nRandomForest Operations (node checks): {ops}\n")

        for path in glob(os.path.join(TEST_DIR, '*.tif')):
            with rasterio.open(path) as src:
                img = src.read([1,2,3,4]).astype(np.float32)
                id_ = os.path.splitext(os.path.basename(path))[0]
            features = img.reshape(4, -1).T
            mask = rf.predict(features).reshape(src.height, src.width)
            writer.writerow([id_, rle.rle_encode(mask)])

    print(f'Results in {OUT_CSV}, model summary in {LOG_FILE}')

if __name__ == '__main__':
    main()

