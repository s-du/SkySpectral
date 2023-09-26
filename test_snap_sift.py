import cv2

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene
from PySide6.QtGui import QPixmap, QImage, QCursor
from PySide6.QtCore import QPoint
import numpy as np

# Load the image
image_path = r'D:\Python2023\Multispectrall\000\ALIGNED_IMG_0032\aligned_2.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints
keypoints = sift.detect(image, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

app = QApplication([])
window = QMainWindow()
scene = QGraphicsScene()
view = QGraphicsView(scene, window)
view.setMouseTracking(True)

# Convert the image with keypoints to QPixmap and display
height, width, channel = image_with_keypoints.shape
bytesPerLine = 3 * width
qimage = QImage(image_with_keypoints.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
qpixmap = QPixmap.fromImage(qimage)
scene.addPixmap(qpixmap)


def snap_to_keypoint(event):
    # Get mouse cursor position
    x, y = event.pos().x(), event.pos().y()

    # Define a search radius
    radius = 5

    nearest_keypoint_pos = None
    min_distance = float('inf')

    # Search for keypoints within the radius
    for kp in keypoints:
        kp_x, kp_y = int(kp.pt[0]), int(kp.pt[1])
        distance = (kp_x - x) ** 2 + (kp_y - y) ** 2
        if distance < min_distance and distance < radius ** 2:
            min_distance = distance
            nearest_keypoint_pos = (kp_x, kp_y)

    # If a keypoint was found within the radius and the distance is significant, snap to it
    if nearest_keypoint_pos and min_distance > 2:
        global_pos = view.mapToGlobal(QPoint(*nearest_keypoint_pos))
        QCursor.setPos(global_pos)


view.mouseMoveEvent = snap_to_keypoint
window.setCentralWidget(view)
window.show()

app.exec()