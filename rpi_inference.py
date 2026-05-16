#!/usr/bin/env python3
"""RPi5 TFLite inference"""
import argparse, time, csv
import numpy as np
from pathlib import Path

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter

T_PIX = 0.50
T_IMG = 0.70
IGNORE_LABEL = 99.0

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

# load model
interp = Interpreter(model_path=f"models/{args.model}", num_threads=args.threads)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

# load patches
xs = sorted(Path("test_npy").glob("x_*.npy"))
ys = sorted(Path("test_npy").glob("y_*.npy"))
n = len(xs)
print(f"{n} patches, model: {args.model}")

# pixel-level accumulators 
TP = FP = TN = FN = 0

# tile-level accumulators 
tTP = tFP = tTN = tFN = 0

lats = []
rows = []

for i in range(n):
    x = np.load(xs[i]).astype(np.float32)[np.newaxis]
    y = np.load(ys[i]).astype(np.float32)  # (512,512,1)

    interp.set_tensor(inp["index"], x)
    t0 = time.perf_counter()
    interp.invoke()
    dt = (time.perf_counter() - t0) * 1000
    lats.append(dt)

    prob = interp.get_tensor(out["index"])  # (1,512,512,1)

    # pixel level 
    p = prob[0, :, :, 0]
    y_ = y[:, :, 0]
    mask = (y_ != IGNORE_LABEL)
    pred = (p >= T_PIX).astype(np.float32)
    m = mask.astype(np.float32)
    y_valid = y_ * m
    p_valid = pred * m

    TP += float(np.sum(y_valid * p_valid))
    FP += float(np.sum((1 - y_valid) * p_valid * m))
    TN += float(np.sum((1 - y_valid) * (1 - p_valid) * m))
    FN += float(np.sum(y_valid * (1 - p_valid) * m))

    # tile level 
    valid = (y_ != IGNORE_LABEL)
    pred_cf = (p[valid] >= T_PIX).mean()
    gt_cf = y_[valid].mean()
    pred_tile = int(pred_cf >= T_IMG)
    gt_tile = int(gt_cf >= T_IMG)

    if   gt_tile == 1 and pred_tile == 1: tTP += 1
    elif gt_tile == 0 and pred_tile == 1: tFP += 1
    elif gt_tile == 1 and pred_tile == 0: tFN += 1
    else:                                 tTN += 1

    rows.append([xs[i].stem, f"{dt:.1f}", f"{pred_cf:.4f}", f"{gt_cf:.4f}", pred_tile, gt_tile])
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n}  last={dt:.0f}ms")

# pixel metrics 
eps = 1e-8
cloud_iou = TP / (TP + FP + FN + eps)
clear_iou = TN / (TN + FP + FN + eps)
prec = TP / (TP + FP + eps)
rec = TP / (TP + FN + eps)
f1 = 2 * TP / (2 * TP + FP + FN + eps)
fpr = FP / (FP + TN + eps)
oa = (TP + TN) / (TP + FP + TN + FN + eps)

# tile metrics 
t_fpr = tFP / (tFP + tTN + eps)
t_de = tTP / (tTP + tFN + eps)

la = np.array(lats)

print(f"\n{'='*60}")
print(f"  {args.name}  ({args.model})")
print(f"  t_pix={T_PIX}  t_img={T_IMG}")
print(f"{'='*60}")
print(f"  Latency: {la.mean():.1f} +/- {la.std():.1f} ms  median={np.median(la):.1f}")
print(f"  Total:   {la.sum()/1000:.1f} s")
print(f"  Pixel:  OA={oa:.4f}  F1={f1:.4f}  mIoU={0.5*(cloud_iou+clear_iou):.4f}")
print(f"          Prec={prec:.4f}  Rec={rec:.4f}  FPR={fpr:.4f}")
print(f"          TP={int(TP)} FP={int(FP)} TN={int(TN)} FN={int(FN)}")
print(f"  Tile:   FPR={t_fpr:.4f}  DL_Eff={t_de:.4f}")
print(f"          TP={tTP} FP={tFP} TN={tTN} FN={tFN}")
print(f"{'='*60}")

with open(f"results_{args.name}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patch", "lat_ms", "pred_cf", "gt_cf", "pred_cloudy", "gt_cloudy"])
    w.writerows(rows)
print("Done.")
