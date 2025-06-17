import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.models import alexnet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns


# === Настройки ===
BASE_DIR = "test_results_2025-06-15_15-22-40"
REAL_DIR = f"{BASE_DIR}/real"
SYNTH_DIR = f"{BASE_DIR}/synth"
REAL_SEG = f"{BASE_DIR}/real_seg"
SYNTH_SEG = f"{BASE_DIR}/synth_seg"
os.makedirs("metrics_summary", exist_ok=True)
LOG_FILE = "metrics_summary/metrics_summary3.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9

COLOR_MAP = np.array([
    [0, 0, 0], [135, 206, 235], [128, 64, 128], [70, 70, 70],
    [34, 139, 34], [160, 82, 45], [107, 142, 35], [190, 153, 153],
    [0, 0, 128], [0, 191, 255]
], dtype=np.uint8)

# === Метрики сегментации ===
def rgb_to_class(img, color_map, threshold=3):
    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    for idx, color in enumerate(color_map):
        dist = np.linalg.norm(img - color, axis=-1)
        mask[dist <= threshold] = idx
    return mask

def compute_segmentation_metrics(gt_list, pred_list, num_classes=10):
    global hist
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    for gt, pred in zip(gt_list, pred_list):
        if gt.shape != pred.shape:
            continue
        valid = (gt != 255) & (pred != 255)
        gt_valid, pred_valid = gt[valid], pred[valid]
        hist += np.bincount(num_classes * gt_valid + pred_valid, minlength=num_classes**2).reshape(num_classes, num_classes)
    acc = np.diag(hist).sum() / (hist.sum() + EPS)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + EPS)
    valid = (hist.sum(1) + hist.sum(0)) > 0
    miou = np.nanmean(iou[valid])
    return acc, miou, iou

def visualize_segmentation_errors(gt, pred, class_idx=2):
    tp = (gt == class_idx) & (pred == class_idx)
    fp = (gt != class_idx) & (pred == class_idx)
    fn = (gt == class_idx) & (pred != class_idx)
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis[tp], vis[fp], vis[fn] = [0, 255, 0], [255, 0, 0], [0, 0, 255]
    return Image.fromarray(vis)

# === Модель сцен ===
def load_places365_model():
    model = alexnet(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 365)
    ckpt = torch.hub.load_state_dict_from_url("http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar")["state_dict"]
    model.load_state_dict({k.replace("features.module.", "features."): v for k, v in ckpt.items()})
    return model.to(DEVICE).eval()

# === Преобразования ===
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(path):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

# === Метрики изображения ===
def compute_ssim(img1, img2):
    return ssim(np.array(Image.open(img1).convert("L")), np.array(Image.open(img2).convert("L")))

def compute_psnr(img1, img2):
    return psnr(np.array(Image.open(img1)), np.array(Image.open(img2)))

def compute_l1(img1, img2):
    a = np.array(Image.open(img1).convert("L")).astype(np.float32) / 255.
    b = np.array(Image.open(img2).convert("L")).astype(np.float32) / 255.
    return np.abs(a - b).mean()

# === Основная функция ===
def evaluate():
    global hist  # чтобы hist можно было использовать в визуализации

    model = load_places365_model()
    files = sorted(os.listdir(REAL_DIR))
    base_names = [os.path.splitext(f)[0] for f in files]

    top1, top5, top1_conf, top5_conf, conf_total = 0, 0, 0, 0, 0
    ssim_list, psnr_list, l1_list = [], [], []
    kl_list, kl_top1_list, kl_top5_list = [], [], []
    seg_gt, seg_pred = [], []

    for name in base_names:
        real_mask = rgb_to_class(np.array(Image.open(f"{REAL_SEG}/{name}.png")), COLOR_MAP)
        synth_mask = rgb_to_class(np.array(Image.open(f"{SYNTH_SEG}/{name}.png")), COLOR_MAP)
        seg_gt.append(real_mask)
        seg_pred.append(synth_mask)
        if name == base_names[0]:
            visualize_segmentation_errors(real_mask, synth_mask).save("debug_road_errors.png")

    acc, miou, ious = compute_segmentation_metrics(seg_gt, seg_pred)

    for fname in tqdm(files):
        real_path, synth_path = f"{REAL_DIR}/{fname}", f"{SYNTH_DIR}/{fname}"
        ssim_list.append(compute_ssim(real_path, synth_path))
        psnr_list.append(compute_psnr(real_path, synth_path))
        l1_list.append(compute_l1(real_path, synth_path))

        real_img, synth_img = load_image(real_path), load_image(synth_path)
        with torch.no_grad():
            real_logits = model(real_img)
            synth_logits = model(synth_img)

        real_probs = F.softmax(real_logits, dim=1).cpu().numpy()[0]
        synth_probs = F.softmax(synth_logits, dim=1).cpu().numpy()[0]
        real_topk = real_probs.argsort()[::-1][:5]
        synth_topk = synth_probs.argsort()[::-1][:5]
        top1 += real_topk[0] == synth_topk[0]
        top5 += real_topk[0] in synth_topk

        if real_probs[real_topk[0]] > 0.5:
            conf_total += 1
            top1_conf += real_topk[0] == synth_topk[0]
            top5_conf += real_topk[0] in synth_topk

        kl_list.append(entropy(synth_probs + EPS, real_probs + EPS))
        mask1 = np.zeros_like(real_probs); mask1[real_topk[0]] = 1
        mask5 = np.zeros_like(real_probs); mask5[real_topk] = 1
        kl_top1_list.append(entropy(synth_probs * mask1 + EPS, real_probs * mask1 + EPS))
        kl_top5_list.append(entropy(synth_probs * mask5 + EPS, real_probs * mask5 + EPS))

    def pct(x, total): return f"{100 * x / total:.2f}%" if total > 0 else "N/A"

    summary = [
        "=== Scene Classification Accuracy ===",
        f"Top-1 Accuracy: {pct(top1, len(files))}",
        f"Top-5 Accuracy: {pct(top5, len(files))}",
        f"Top-1 Accuracy (prob > 0.5): {pct(top1_conf, conf_total)}",
        f"Top-5 Accuracy (prob > 0.5): {pct(top5_conf, conf_total)}",
        "",
        "=== Image Quality Metrics ===",
        f"SSIM: {np.mean(ssim_list):.3f}",
        f"PSNR: {np.mean(psnr_list):.2f} dB",
        f"Sharpness (L1): {np.mean(l1_list):.3f}",
        "",
        "=== KL Divergence ===",
        f"KL(synth || real): {np.mean(kl_list):.3f}",
        f"exp(KL): {np.exp(np.mean(kl_list)):.3f}",
        f"std(KL): {np.std(kl_list):.3f}",
        f"KL_top1: {np.mean(kl_top1_list):.3f}",
        f"KL_top5: {np.mean(kl_top5_list):.3f}",
        "",
        "=== Segmentation Metrics ===",
        f"Pixel Accuracy: {acc:.3f}",
        f"Mean IoU: {miou:.3f}",
        f"Class IoU: {[f'{v:.2f}' for v in ious]}"
    ]

    print("\n".join(summary))
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    os.makedirs("test_plots", exist_ok=True)

    # === Визуализация confusion matrix ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(hist, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("test_plots/confusion_matrix.png")
    plt.close()

    # === Визуализация per-class IoU ===
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(ious)), ious, color='skyblue')
    plt.xlabel("Class Index")
    plt.ylabel("IoU")
    plt.title("Per-class IoU")
    plt.ylim(0, 1)
    plt.xticks(range(len(ious)))
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("test_plots/iou_per_class.png")
    plt.close()

    # === Визуализация распределения KL-дивергенции ===
    plt.figure(figsize=(8, 4))
    plt.hist(kl_list, bins=30, color='coral', alpha=0.8)
    plt.axvline(np.mean(kl_list), color='black', linestyle='--', label=f"Mean KL = {np.mean(kl_list):.3f}")
    plt.title("KL Divergence Distribution")
    plt.xlabel("KL(synth || real)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_plots/kl_distribution.png")
    plt.close()


if __name__ == "__main__":
    evaluate()
