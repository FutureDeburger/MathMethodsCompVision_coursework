import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== НАСТРОЙКИ =====================
DATA_DIR = "data"
RESULTS_DIR = "results"

NOISE_LEVELS = [0, 10]
SCALES = [1.0, 0.8]
ROTATIONS = [0, 15]

DETECTORS = ["SIFT", "ORB", "AKAZE"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/plots", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/panoramas", exist_ok=True)

# ===================== ЗАГРУЗКА ДАННЫХ =====================
def load_images():
    images = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.lower().endswith(".jpg"):
            img = cv2.imread(os.path.join(DATA_DIR, f))
            if img is not None:
                images.append(img)
    return images

# ===================== ИСКАЖЕНИЯ =====================
def add_noise(img, level):
    if level == 0:
        return img.copy()
    noise = np.random.normal(0, level, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def scale_image(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def generate_variations(img):
    out = []
    for n in NOISE_LEVELS:
        for s in SCALES:
            for r in ROTATIONS:
                v = rotate_image(scale_image(add_noise(img, n), s), r)
                out.append((v, n, s, r))
    return out

# ===================== ДЕТЕКТОРЫ =====================
def get_detector(name):
    if name == "SIFT":
        return cv2.SIFT_create()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=2000)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError("Unknown detector")

def run_detector(detector, img):
    t0 = time.time()
    kp, des = detector.detectAndCompute(img, None)
    return kp, des, time.time() - t0

# ===================== MATCHING =====================
def match_descriptors(des1, des2, name):
    if des1 is None or des2 is None:
        return []
    if len(des1) < 2 or len(des2) < 2:
        return []

    norm = cv2.NORM_HAMMING if name in ["ORB", "AKAZE"] else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)

    try:
        return matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []

def lowe_ratio(matches, ratio=0.75):
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    return good

# ===================== ПАНОРАМА =====================
def create_panorama(img1, img2, kp1, kp2, matches):
    if len(matches) < 10:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None or np.sum(mask) < 10:
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pano = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    pano[0:h2, 0:w2] = img2
    return pano

# ===================== ОСНОВНОЙ ЗАПУСК =====================
if __name__ == "__main__":
    images = load_images()
    if len(images) < 2:
        raise RuntimeError("Нужно минимум 2 изображения в папке data")

    metrics = []

    for name in DETECTORS:
        print(f"\n=== {name} ===")
        detector = get_detector(name)

        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                vars1 = generate_variations(images[i])
                vars2 = generate_variations(images[j])

                for v1, n1, s1, r1 in vars1:
                    kp1, des1, t1 = run_detector(detector, v1)

                    for v2, n2, s2, r2 in vars2:
                        kp2, des2, t2 = run_detector(detector, v2)

                        matches = match_descriptors(des1, des2, name)
                        good = lowe_ratio(matches)

                        metrics.append({
                            "detector": name,
                            "kp1": len(kp1),
                            "kp2": len(kp2),
                            "good_matches": len(good),
                            "noise1": n1,
                            "noise2": n2,
                            "scale1": s1,
                            "scale2": s2,
                            "rot1": r1,
                            "rot2": r2,
                            "time": t1 + t2
                        })

                        # Панорамы только для чистых условий
                        if n1 == n2 == 0 and r1 == r2 == 0:
                            pano = create_panorama(v1, v2, kp1, kp2, good)
                            if pano is not None:
                                cv2.imwrite(
                                    f"{RESULTS_DIR}/panoramas/{name}_{i}_{j}.jpg",
                                    pano
                                )

    # ===================== СОХРАНЕНИЕ =====================
    df = pd.DataFrame(metrics)
    df.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
    print("\nМетрики сохранены")

    # ===================== ГРАФИКИ =====================
    for name in DETECTORS:
        sub = df[df["detector"] == name]
        if sub.empty:
            continue

        plt.figure(figsize=(7, 5))
        plt.scatter(sub["kp1"] + sub["kp2"], sub["good_matches"], alpha=0.4)
        plt.xlabel("Количество ключевых точек")
        plt.ylabel("Хорошие совпадения")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/plots/{name}_matches.png")
        plt.close()

    print("Графики построены. Эксперимент завершён.")
