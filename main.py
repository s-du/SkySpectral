import sys
import os
import cv2

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader


def createLineIterator(P1, P2, img):
    """
    Source: https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(int), itbuffer[:, 0].astype(int)]

    return itbuffer


class PhotoViewer(QGraphicsView):
    photoClicked = Signal(QPoint)
    endDrawing_rect = Signal()
    end_point_selection = Signal()
    endDrawing_line_meas = Signal()

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setFrameShape(QFrame.NoFrame)

        self.rect = False
        self.select_point = False
        self.line_meas = False

        self.setMouseTracking(True)
        self.origin = QPoint()

        self._current_rect_item = None
        self._current_line_item = None
        self._current_point = None
        self._current_path = None

        self.crop_coords = []

        self.pen = QPen()
        self.pen.setStyle(Qt.DashDotLine)
        self.pen.setWidth(4)
        self.pen.setColor(QColor(255, 0, 0, a=255))
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setJoinStyle(Qt.RoundJoin)

        self.meas_color = QColor(0, 100, 255, a=255)
        self.pen_yolo = QPen()
        # self.pen.setStyle(Qt.DashDotLine)
        self.pen_yolo.setWidth(2)
        self.pen_yolo.setColor(self.meas_color)
        self.pen_yolo.setCapStyle(Qt.RoundCap)
        self.pen_yolo.setJoinStyle(Qt.RoundJoin)

    def has_photo(self):
        return not self._empty

    def showEvent(self, event):
        self.fitInView()
        super(PhotoViewer, self).showEvent(event)

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        print(rect)
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                print('unity: ', unity)
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                print('view: ', viewrect)
                scenerect = self.transform().mapRect(rect)
                print('scene: ', viewrect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def clean_scene(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsRectItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsTextItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsPolygonItem):
                self._scene.removeItem(item)

    def clean_scene_line(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsLineItem):
                self._scene.removeItem(item)

    def clean_scene_rectangle(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsRectItem):
                self._scene.removeItem(item)

    def clean_scene_poly(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsPolygonItem):
                self._scene.removeItem(item)

    def clean_scene_text(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsTextItem):
                self._scene.removeItem(item)

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def toggleDragMode(self):
        if self.rect or self.select_point:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            if self.dragMode() == QGraphicsView.ScrollHandDrag:
                self.setDragMode(QGraphicsView.NoDrag)
            elif not self._photo.pixmap().isNull():
                self.setDragMode(QGraphicsView.ScrollHandDrag)

    def add_poly(self, coordinates):
        # Create a QPolygonF from the coordinates
        polygon = QPolygonF()
        for x, y in coordinates:
            polygon.append(QPointF(x, y))

        # Create a QGraphicsPolygonItem and set its polygon
        polygon_item = QGraphicsPolygonItem(polygon)
        fill_color = QColor(0, 255, 255, 100)
        polygon_item.setBrush(fill_color)  # Set fill color

        # Add the QGraphicsPolygonItem to the scene
        self._scene.addItem(polygon_item)

    def add_yolo_box(self, text, x1, y1, x2, y2):
        # add box
        box = QGraphicsRectItem()
        box.setPen(self.pen_yolo)

        r = QRectF(x1, y1, x2 - x1, y2 - y1)
        box.setRect(r)

        # add text
        text_item = QGraphicsTextItem()
        text_item.setPos(x1, y1)
        text_item.setHtml(
            "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + text + "</div>")

        # add elements to scene
        self._scene.addItem(box)
        self._scene.addItem(text_item)

    def add_list_poly(self, list_objects):
        for el in list_objects:
            # Create a QPolygonF from the coordinates
            polygon = QPolygonF()
            for x, y in el.coords:
                polygon.append(QPointF(x, y))

            # Create a QGraphicsPolygonItem and set its polygon
            polygon_item = QGraphicsPolygonItem(polygon)
            fill_color = QColor(0, 255, 255, 100)
            polygon_item.setBrush(fill_color)  # Set fill color

            # Add the QGraphicsPolygonItem to the scene
            self._scene.addItem(polygon_item)

    def add_list_infos(self, list_objects, only_name=False):
        for el in list_objects:
            x1, y1, x2, y2, score, class_id = el.yolo_bbox
            text = el.name
            text2 = str(round(el.area, 2)) + 'm² '
            text3 = str(round(el.volume, 2)) + 'm³'

            print(f'adding {text} to viewer')

            # add text 1
            text_item = QGraphicsTextItem()
            text_item.setPos(x1, y1)
            text_item.setHtml(
                "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + text + "</div>")

            self._scene.addItem(text_item)

            if not only_name:
                # add text 2 and 3
                text_item2 = QGraphicsTextItem()
                text_item2.setPos(x1, y2)
                text_item2.setHtml(
                    "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + text2 + "<br>" + text3 + " </div>")
                self._scene.addItem(text_item2)

    def add_list_boxes(self, list_objects):
        for el in list_objects:
            x1, y1, x2, y2, score, class_id = el.yolo_bbox

            # add box
            box = QGraphicsRectItem()
            box.setPen(self.pen_yolo)

            r = QRectF(x1, y1, x2 - x1, y2 - y1)
            box.setRect(r)

            # add elements to scene
            self._scene.addItem(box)

    def get_coord(self, QGraphicsRect):
        rect = QGraphicsRect.rect()
        coord = [rect.topLeft(), rect.bottomRight()]
        print(coord)

        return coord

    def get_selected_point(self):
        print(self._current_point)
        return self._current_point

    def set_height_data(self, height_data):
        self.height_values = height_data

    # mouse events
    def wheelEvent(self, event):
        print(self._zoom)
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def mousePressEvent(self, event):
        if self.rect:
            self._current_rect_item = QGraphicsRectItem()
            self._current_rect_item.setFlag(QGraphicsItem.ItemIsSelectable)
            self._current_rect_item.setPen(self.pen)
            self._scene.addItem(self._current_rect_item)
            self.origin = self.mapToScene(event.pos())
            r = QRectF(self.origin, self.origin)
            self._current_rect_item.setRect(r)

        elif self.select_point:
            self._current_point = self.mapToScene(event.pos())
            self.get_selected_point()
            self.select_point = False
            self.end_point_selection.emit()

        elif self.line_meas:
            self._current_line_item = QGraphicsLineItem()
            self._current_line_item.setPen(self.pen_yolo)

            self._scene.addItem(self._current_line_item)
            self.origin = self.mapToScene(event.pos())

            self._current_line_item.setLine(QLineF(self.origin, self.origin))

        else:
            if self._photo.isUnderMouse():
                self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rect:
            if self._current_rect_item is not None:
                new_coord = self.mapToScene(event.pos())
                r = QRectF(self.origin, new_coord)
                self._current_rect_item.setRect(r)
        elif self.line_meas:
            if self._current_line_item is not None:
                self.new_coord = self.mapToScene(event.pos())
                self._current_line_item.setLine(QLineF(self.origin, self.new_coord))

        super(PhotoViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect:
            self.rect = False
            self.origin = QPoint()
            if self._current_rect_item is not None:
                self.crop_coords = self.get_coord(self._current_rect_item)
                self.endDrawing_rect.emit()
                print('rectangle ROI added: ' + str(self.crop_coords))
            self._current_rect_item = None
            self.toggleDragMode()

        elif self.line_meas:
            self.line_meas = False

            if self._current_line_item is not None:
                # compute line values
                p1 = np.array([int(self.origin.x()), int(self.origin.y())])
                p2 = np.array([int(self.new_coord.x()), int(self.new_coord.y())])
                print(p1,p2)
                line_values = createLineIterator(p1, p2, self.height_values)

                self.line_values_final = line_values[:,2]

                # emit signal (end of measure)
                self.endDrawing_line_meas.emit()
                print('Line meas. added')

            self.origin = QPoint()
            self._current_line_item = None
            self.toggleDragMode()

        super(PhotoViewer, self).mouseReleaseEvent(event)


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Micasense Rededge P Image Processor")

        # UI Elements
        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image Directory")
        self.load_button.clicked.connect(self.load_images)
        self.layout.addWidget(self.load_button)

        # Shot List and its label
        self.shot_label = QLabel("Select Shot:")
        self.layout.addWidget(self.shot_label)

        self.shot_list = QListWidget()
        self.shot_list.setViewMode(QListWidget.IconMode)
        self.shot_list.setLayoutDirection(Qt.LeftToRight)

        self.shot_list.setIconSize(QSize(64, 64))  # Set the desired icon size
        self.shot_list.itemClicked.connect(self.shot_selected)
        self.layout.addWidget(self.shot_list)

        # Band ComboBox and its label
        self.band_label = QLabel("Select Band:")
        self.layout.addWidget(self.band_label)

        self.band_combobox = QComboBox()
        self.band_combobox.addItems(['blue', 'green', 'red', 'NIR', 'red edge', 'panchromatic'])
        self.band_combobox.currentIndexChanged.connect(self.update_display)
        self.layout.addWidget(self.band_combobox)

        # Palette ComboBox and its label
        self.palette_label = QLabel("Select Color Palette:")
        self.layout.addWidget(self.palette_label)

        self.palette_combobox = QComboBox()
        self.palettes = sorted(plt.colormaps())
        self.palette_combobox.addItems(self.palettes)
        self.palette_combobox.currentIndexChanged.connect(self.update_display)
        self.layout.addWidget(self.palette_combobox)

        self.imageviewer = PhotoViewer(self)
        self.layout.addWidget(self.imageviewer)


        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Variables for our images
        self.base_dir = ""
        self.shots = []

    def load_images(self):
        self.base_dir = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not self.base_dir:
            return

        # Find unique shots in the directory
        filenames = os.listdir(self.base_dir)
        self.shots = sorted(set(name.split('_')[1] for name in filenames if name.endswith('.tif')))

        # Populate shot list with icons
        for shot in self.shots:
            band_1_path = os.path.join(self.base_dir, f"IMG_{shot}_1.tif")  # Assuming 1st band for thumbnail
            thumbnail = self.generate_thumbnail(band_1_path)
            item = QListWidgetItem(QIcon(thumbnail), shot)
            self.shot_list.addItem(item)

        self.update_display()

    def generate_thumbnail(self, img_path):
        thumbnail_size = (64, 64)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, thumbnail_size, interpolation=cv2.INTER_AREA)
        height, width = image.shape
        q_img = QImage(image.data, width, height, QImage.Format_Grayscale16)
        pixmap = QPixmap.fromImage(q_img)
        return pixmap

    def shot_selected(self, item):
        # Load and display the shot corresponding to the clicked item
        self.selected_shot = item.text()
        self.update_display()

    def update_display(self):
        shot = self.selected_shot
        band = self.band_combobox.currentIndex() + 1

        if not shot:
            return

        filepath = os.path.join(self.base_dir, f"IMG_{shot}_{band}.tif")
        print(filepath)

        # Using OpenCV to load image
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        print(image.dtype)

        # Get the selected colormap
        selected_cmap = self.palette_combobox.currentText()
        cmap = cm.get_cmap(selected_cmap)

        # Convert to float for proper division
        image_float = image.astype(np.float32)

        # Normalize to 0-1 range
        norm_image = (image_float - image_float.min()) / (image_float.max() - image_float.min())

        # Ensure there aren't any NaN values due to division by zero
        norm_image = np.nan_to_num(norm_image)

        # Apply colormap (this gives an RGBA image)
        colored_image = (cm.get_cmap(selected_cmap)(norm_image) * 255).astype(np.uint8)

        height, width, _ = colored_image.shape
        q_img = QImage(colored_image.data, width, height, QImage.Format_RGBA8888)


        pixmap = QPixmap.fromImage(q_img)
        self.imageviewer.setPhoto(pixmap=pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.showMaximized()
    sys.exit(app.exec())