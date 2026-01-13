import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== НАСТРОЙКИ =====================
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
PANOS_DIR = os.path.join(RESULTS_DIR, "panoramas")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PANOS_DIR, exist_ok=True)

DETECTORS = ["SIFT", "ORB", "AKAZE"]

# ===================== ДЕТЕКТОРЫ =====================
def get_detector(name):
    if name == "SIFT":
        return cv2.SIFT_create()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(name)

def get_matcher(name):
    if name == "SIFT":
        return cv2.BFMatcher(cv2.NORM_L2)
    return cv2.BFMatcher(cv2.NORM_HAMMING)

# ===================== ЗАГРУЗКА ДАННЫХ =====================
def load_images():
    images = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.lower().endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(DATA_DIR, f))
            if img is not None:
                images.append((f, img))
    return images

# ===================== МАТЧИНГ =====================
def match_keypoints(detector_name, img1, img2):
    detector = get_detector(detector_name)
    matcher = get_matcher(detector_name)

    t0 = time.time()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    t_detect = time.time() - t0

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return {
        "kp1": kp1,
        "kp2": kp2,
        "good": good,
        "time": t_detect
    }

# ===================== ПАНОРАМА =====================
def build_panorama(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pano = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    pano[0:h2, 0:w2] = img2
    return pano

# ===================== ОСНОВНОЙ ЭКСПЕРИМЕНТ =====================
def run():
    images = load_images()
    if len(images) < 2:
        raise RuntimeError("Нужно минимум 2 изображения")

    metrics = []

    for det in DETECTORS:
        print(f"\n=== {det} ===")

        for i in range(len(images) - 1):
            name1, img1 = images[i]
            name2, img2 = images[i + 1]

            result = match_keypoints(det, img1, img2)
            if result is None:
                continue

            pano = build_panorama(
                img1, img2,
                result["kp1"],
                result["kp2"],
                result["good"]
            )

            metrics.append({
                "detector": det,
                "image_pair": f"{name1}-{name2}",
                "kp1": len(result["kp1"]),
                "kp2": len(result["kp2"]),
                "good_matches": len(result["good"]),
                "time": result["time"]
            })

            if pano is not None:
                out = f"{det}_{name1}_{name2}.jpg"
                cv2.imwrite(os.path.join(PANOS_DIR, out), pano)

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    plot_results(df)
    print("\nГотово. Результаты в папке results/")

# ===================== ГРАФИКИ =====================
def plot_results(df):
    for det in df.detector.unique():
        sub = df[df.detector == det]

        plt.figure(figsize=(7, 5))
        plt.scatter(sub["kp1"] + sub["kp2"], sub["good_matches"])
        plt.xlabel("Число ключевых точек")
        plt.ylabel("Хорошие совпадения")
        plt.title(det)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{det}_matches.png"))
        plt.close()

# ===================== ЗАПУСК =====================
if __name__ == "__main__":
    print("OpenCV:", cv2.__version__)
    run()
