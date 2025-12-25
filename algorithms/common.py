import cv2
import matplotlib.pyplot as plt
import time

def load_image(path, gray=True):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Не удалось загрузить изображение: {path}")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def match_descriptors(des1, des2, detector_name):
    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    if detector_name in ["ORB", "AKAZE"]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(des1, des2, k=2)
    return matches


def lowe_ratio_test(matches, ratio=0.75):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50, save_path=None):
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(15, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def detect_and_compute(detector, img):
    start = time.time()
    kp, des = detector.detectAndCompute(img, None)
    elapsed = time.time() - start
    return kp, des, elapsed
