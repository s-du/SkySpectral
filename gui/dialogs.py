import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from gui import widgets as wid
import os
import numpy as np

class ShowComposed(QDialog):
    def __init__(self, img_paths, img_names, parent=None):
        QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'composed'
        uifile = os.path.join(basepath, 'gui/ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        # combobox
        self.img_paths = img_paths
        self.comboBox_views.addItems(img_names)

        self.viewer = wid.PhotoViewer(self)
        self.verticalLayout.addWidget(self.viewer)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.comboBox_views.currentIndexChanged.connect(self.on_img_combo_change)

    def on_img_combo_change(self):
        i = self.comboBox_views.currentIndex()
        img_path = self.img_paths[i]
        self.viewer.setPhoto(QPixmap(img_path))


class AlignmentWindowArrow(QDialog):
    def __init__(self, ref_path, targ_path):
        super().__init__()

        # Load images
        self.reference_image = QImage(ref_path)
        self.to_align_image = QImage(targ_path)
        self.x_offset = 0
        self.y_offset = 0

        # Initialize the GUI components
        self.init_ui()

        # Display the images
        self.display_images()

    def init_ui(self):
        # Main Layout
        layout = QVBoxLayout()

        # Image Display
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Joystick buttons
        joystick_layout = QHBoxLayout()
        self.left_button = QPushButton("Left", self)
        self.right_button = QPushButton("Right", self)
        self.up_button = QPushButton("Up", self)
        self.down_button = QPushButton("Down", self)
        joystick_layout.addWidget(self.left_button)
        joystick_layout.addWidget(self.up_button)
        joystick_layout.addWidget(self.down_button)
        joystick_layout.addWidget(self.right_button)
        layout.addLayout(joystick_layout)

        # Connect joystick buttons to their functions
        self.left_button.clicked.connect(self.move_left)
        self.right_button.clicked.connect(self.move_right)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)

        # ComboBox for selecting pixel movement
        self.pixel_selector = QComboBox(self)
        self.pixel_selector.addItems(['1 pixel', '10 pixels', '20 pixels'])
        layout.addWidget(self.pixel_selector)

        self.setLayout(layout)

    def display_images(self):
        # Create an image for blending
        blended_image = QImage(self.reference_image.size(), QImage.Format_ARGB32)
        blended_image.fill(QColor(0, 0, 0, 0))

        # Painter for blending
        painter = QPainter(blended_image)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawImage(0, 0, self.reference_image)
        painter.setCompositionMode(QPainter.CompositionMode_Multiply)
        painter.drawImage(self.x_offset, self.y_offset, self.to_align_image)
        painter.end()

        # Set the blended image to label
        self.image_label.setPixmap(QPixmap.fromImage(blended_image))

    def move_image(self, dx, dy):
        transform = QTransform()
        transform.translate(dx, dy)
        self.to_align_image = self.to_align_image.transformed(transform)
        self.display_images()

    def move_left(self):
        move_amount = self.get_move_amount()
        self.x_offset -= move_amount
        self.display_images()

    def move_right(self):
        move_amount = self.get_move_amount()
        self.x_offset += move_amount
        self.display_images()

    def move_up(self):
        move_amount = self.get_move_amount()
        self.y_offset -= move_amount
        self.display_images()

    def move_down(self):
        move_amount = self.get_move_amount()
        self.y_offset += move_amount
        self.display_images()

    def get_move_amount(self):
        move_str = self.pixel_selector.currentText().split()[0]
        return int(move_str)


class AlignmentWindow(QDialog):
    def __init__(self, ref_img, target_img):
        super().__init__()
        self.setWindowTitle("Manual Image Alignment")
        self.showMaximized()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.ref_img = QImage(ref_img)
        self.target_img = QImage(target_img)

        self.ref_img_path = ref_img
        self.target_img_path = target_img

        self.ref_points = []
        self.target_points = []

        self.ref_view = self.create_graphics_view(self.ref_img)
        self.target_view = self.create_graphics_view(self.target_img)

        self.layout.addWidget(self.ref_view)
        self.layout.addWidget(self.target_view)

        # Controls
        control_layout = QVBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        control_layout.addWidget(self.ok_button)
        self.layout.addLayout(control_layout)

    def create_graphics_view(self, img):
        scene = QGraphicsScene()
        view = wid.ClickableGraphicsView(scene)
        pixmap = QPixmap.fromImage(img)
        scene.addPixmap(pixmap)
        scene.setSceneRect(pixmap.rect())
        view.pointClicked.connect(lambda point, s=scene: self.on_image_click(point, s))
        return view

    def load_points(self, point_list):
        self.ref_points = point_list
        for i, point in enumerate(point_list):
            self.add_coord_marker(self.ref_view.scene(), point, i+1)

    def on_image_click(self, point, scene):
        # point is now directly in scene's coordinates
        if len(self.ref_points) < 6 and scene == self.ref_view.scene():
            self.ref_points.append((point.x(), point.y()))
            self.add_point_marker(scene, point, len(self.ref_points))
        elif len(self.target_points) < 6 and scene == self.target_view.scene():
            self.target_points.append((point.x(), point.y()))
            self.add_point_marker(scene, point, len(self.target_points))

    def add_coord_marker(self, scene, point, number):
        color = QColor(Qt.cyan)
        scene.addEllipse(point[0] - 5, point[1] - 5, 10, 10, QPen(color), color)
        text_item = scene.addText(str(number))
        text_item.setDefaultTextColor(Qt.cyan)
        text_item.setPos(point[0] + 10, point[1] - 10)

        font = QFont()
        font.setPointSize(15)  # Change 20 to your desired font size
        text_item.setFont(font)

    def add_point_marker(self, scene, point, number):
        color = QColor(Qt.cyan)
        scene.addEllipse(point.x() - 5, point.y() - 5, 10, 10, QPen(color), color)
        text_item = scene.addText(str(number))
        text_item.setDefaultTextColor(Qt.cyan)
        text_item.setPos(point.x() + 10, point.y() - 10)

        font = QFont()
        font.setPointSize(15)  # Change 20 to your desired font size
        text_item.setFont(font)

    def get_aligned_image(self):
        # Compute homography using the selected points
        print(np.array(self.target_points))
        print(np.array(self.ref_points))
        rigid = False

        # Estimate the rigid transformation
        if rigid:
            M, _ = cv2.estimateAffinePartial2D(np.array(self.target_points), np.array(self.ref_points), method=cv2.RANSAC)

            # Convert the 2x3 matrix to 3x3 to use with warpPerspective
            H = np.vstack([M, [0, 0, 1]])

        else:
            H, _ = cv2.estimateAffine2D(np.array(self.target_points), np.array(self.ref_points), method=cv2.RANSAC)

        # Warp the target image to the reference image
        target_img = cv2.imread(self.target_img_path)
        aligned = cv2.warpAffine(target_img, H, (self.ref_img.width(), self.ref_img.height()))

        return aligned
