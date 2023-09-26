import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene
from PySide6.QtGui import QPixmap, QImage, QCursor
from PySide6.QtCore import Qt, QPointF

# Load the image and apply edge detection
image_path = r'D:\Python2023\Multispectrall\000\ALIGNED_IMG_0032\aligned_2.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detection with Gaussian blur
img_blur = cv2.GaussianBlur(image, (5, 5), 0)
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_DEFAULT)

grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
sobel_edges = (grad * 255 / grad.max()).astype(np.uint8)
ret, binary_edges = cv2.threshold(sobel_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('RGB Image', binary_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

app = QApplication([])
window = QMainWindow()
scene = QGraphicsScene()
view = QGraphicsView(scene, window)
view.setMouseTracking(True)

# Convert the original image to QPixmap and display
qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
qpixmap = QPixmap.fromImage(qimage)
scene.addPixmap(qpixmap)


def snap_to_edge(event):
    # Map mouse position from view coordinates to scene coordinates
    scene_pos = view.mapToScene(event.pos())

    # Convert scene position to image pixel coordinates
    x = int(scene_pos.x())
    y = int(scene_pos.y())

    print(x,y)

    # Define a search radius
    radius = 10

    nearest_edge_pos = None
    min_distance = float('inf')

    # Search for edges within the radius
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if 0 <= x + i < sobel_edges.shape[1] and 0 <= y + j < sobel_edges.shape[0]:
                if sobel_edges[y + j, x + i] > 30:
                    distance = i ** 2 + j ** 2
                    if distance < min_distance:
                        min_distance = distance
                        nearest_edge_pos = (x + i, y + j)

    # If an edge was found within the radius and the distance is significant, snap to it
    if nearest_edge_pos and min_distance > 2:  # Added a threshold for snapping
        # Convert image pixel coordinates back to scene coordinates
        scene_edge_pos = QPointF(*nearest_edge_pos)
        # Convert scene coordinates to view coordinates
        view_edge_pos = view.mapFromScene(scene_edge_pos)
        # Convert view coordinates to global screen coordinates
        global_pos = view.mapToGlobal(view_edge_pos)
        QCursor.setPos(global_pos)


view.mouseMoveEvent = snap_to_edge
window.setCentralWidget(view)
window.show()

app.exec()