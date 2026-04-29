import cv2


def list_cameras(max_tested=10):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


print(list_cameras())
