import cv2
import os
import time
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_PANO = "results/panoramas"
OUT_METRICS = "results/metrics"

os.makedirs(OUT_PANO, exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)

def load_images():
    imgs = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.lower().endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(DATA_DIR, f))
            if img is not None:
                imgs.append((f, img))
    return imgs

def build_panorama(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pano = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    pano[0:h2, 0:w2] = img2
    return pano

def main():
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    images = load_images()
    metrics = []

    for i in range(len(images)-1):
        name1, img1 = images[i]
        name2, img2 = images[i+1]

        t0 = time.time()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        t = time.time() - t0

        if des1 is None or des2 is None:
            continue

        matches = matcher.knnMatch(des1, des2, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]

        pano = build_panorama(img1, img2, kp1, kp2, good)
        if pano is not None:
            cv2.imwrite(f"{OUT_PANO}/SIFT_{name1}_{name2}.jpg", pano)

        metrics.append({
            "algorithm": "SIFT",
            "pair": f"{name1}-{name2}",
            "kp1": len(kp1),
            "kp2": len(kp2),
            "good_matches": len(good),
            "time": t
        })

    pd.DataFrame(metrics).to_csv(f"{OUT_METRICS}/sift_metrics.csv", index=False)
    print("SIFT: готово")

if __name__ == "__main__":
    main()
