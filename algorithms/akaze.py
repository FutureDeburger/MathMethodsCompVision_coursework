import cv2
from algorithms.common import detect_and_compute

def create_detector():
    return cv2.AKAZE_create()

def run_detector(img):
    detector = create_detector()
    return detect_and_compute(detector, img)
