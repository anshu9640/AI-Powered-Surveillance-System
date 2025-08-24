# train/train_behavior.py
import os, glob, cv2, argparse, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.autoencoder3d import AE3D

# Supported extensions
IMG_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
VID_EXTS = {'.avi', '.mp4', '.mov', '.mkv'}

def list_videos(root):
    paths = []
    for ext in VID_EXTS:
        paths += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
    return paths

def list_img_folders(root, min_images=16):
    folders = set()
    for dirpath, _, filenames in os.walk(root):
        cnt = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if cnt >= min_images:
            folders.add(dirpath)
    return sorted(folders)

def read_video_gray(path):
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened():
        return frames
    ok, frame = cap.read()
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        ok, frame = cap.read()
    cap.release()
    return frames

def read_folder_gray(dirpath):
    files = sorted(
        [p for p in glob.glob(os.path.join(dirpath, "*"))
         if os.path.splitext(p)[1].lower() in IMG_EXTS]
    )
    frames = []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return frames

def resize112(img):
    return cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

class ClipDataset(Dataset):
    """
    Builds (T=16,H=112,W=112) grayscale clips from:
      - video files under root (avi/mp4/...)
      - any subfolder under root that contains >=16 images (tif/png/jpg/...)
    """
    def __init__(self, root, clip_len=16, stride=None):
        self.clips = []
        self.clip_len = clip_len
        self.stride = stride if stride is not None else clip_len  # non-overlap by default

        # 1) videos
        for vp in list_videos(root):
            fs = read_video_gray(vp)
            if len(fs) < clip_len:
                continue
            fs = [resize112(f) for f in fs]
            for i in range(0, len(fs) - clip_len + 1, self.stride):
                self.clips.append(np.stack(fs[i:i + clip_len], axis=0))

        # 2) image-sequence folders
        for folder in list_img_folders(root, min_images=clip_len):
            fs = read_folder_gray(folder)
            if len(fs) < clip_len:
                continue
            fs = [resize112(f) for f in fs]
            for i in range(0, len(fs) - clip_len + 1, self.stride):
                self.clips.append(np.stack(fs[i:i + clip_len], axis=0))

        random.shuffle(self.clips)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx].astype(np.float32) / 255.0  # (T,H,W)
        if random.random() < 0.5:
            c = c[:, :, ::-1].copy()   # <-- this was the syntax error; keep it exactly like this
        c = np.expand_dims(c, axis=0)  # (1,T,H,W)
        return torch.from_numpy(c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True,
                    help="Folder with NORMAL clips: videos or subfolders of images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stride", type=int, default=16, help="Frame stride when slicing clips")
    args = ap.parse_args()

    ds = ClipDataset(args.train_dir, clip_len=16, stride=args.stride)
    n = len(ds)
    print(f"Found {n} clips in '{args.train_dir}'")
    if n == 0:
        raise RuntimeError(
            f"No clips found under {args.train_dir}. "
            "Ensure it contains videos or subfolders with >=16 images."
        )

    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    model = AE3D().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
        ema = None
        for x in pbar:
            x = x.to(args.device)          # (B,1,T,H,W)
            xhat = model(x)
            loss = ((x - xhat) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            pbar.set_postfix(loss=f"{ema:.5f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ae3d.pth")
    print("saved models\\ae3d.pth")

if __name__ == "__main__":
    main()
