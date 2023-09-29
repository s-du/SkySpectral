import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from gui import widgets as wid
import os
import numpy as np
import resources as res
import re
import matplotlib.cm as cm


class RasterTransformDialog(QDialog):
    def __init__(self, images):
        super().__init__()

        # Window title and size
        self.setWindowTitle("Raster Transform")
        # Set the minimum size
        self.setMinimumSize(800, 600)

        # Main layout
        main_layout = QHBoxLayout(self)

        # List of bands
        self.bands_list = QListWidget(self)
        bands = ["Blue (B)", "Green (G)",  "Red (R)", "RedEdge (RE)", "NearInfrared (NIR)"]
        self.bands_list.addItems(bands)
        self.bands_list.itemDoubleClicked.connect(self.add_band_to_formula)
        main_layout.addWidget(self.bands_list)

        self.indices = {
            "NDVI": "(NIR - R) / (NIR + R)",
            "SR": "NIR / R",
            "MSI": "NIR / RE",
            "NDWI": "(G - NIR) / (G + NIR)"
        }

        # Right side layout
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        self.indices_combobox = QComboBox(self)
        self.indices_combobox.addItem("Custom")  # An option for custom formulas
        for index_name in self.indices.keys():
            self.indices_combobox.addItem(index_name)
        right_layout.addWidget(self.indices_combobox)

        self.indices_combobox.currentIndexChanged.connect(self.on_index_changed)

        # Formula input
        self.formula_input = QLineEdit(self)
        right_layout.addWidget(self.formula_input)

        # Formula correctness indicator
        self.formula_indicator = QLabel(self)
        right_layout.addWidget(self.formula_indicator)

        # Operators grid
        operators_grid = QGridLayout()
        right_layout.addLayout(operators_grid)

        operators = [
            "+", "-", "*", "/",
            "asin", "sin", "cos", "acos",
            "tan", "atan", "(", ")",
            "^", "exp", "log", "sqrt"
        ]
        for i, op in enumerate(operators):
            btn = self.create_operator_button(op)
            operators_grid.addWidget(btn, i // 4, i % 4)

        # Connect formula input changes to the check function
        self.formula_input.textChanged.connect(self.check_formula)

        # Load images
        self.images = images

        # Colormap dropdown
        self.palettes = [
            'Greys_r',
            'Greys',
            'Spectral',
            'Spectral_r',
            'afmhot',
            'afmhot_r',
            'bwr',
            'bwr_r',
            'coolwarm',
            'coolwarm_r',
            'gnuplot2',
            'gnuplot2_r',
            'inferno',
            'inferno_r',
            'jet',
            'jet_r',
            'magma',
            'magma_r',
            'nipy_spectral',
            'nipy_spectral_r',
            'plasma',
            'plasma_r',
            'rainbow',
            'rainbow_r',
            'seismic',
            'seismic_r',
            'turbo',
            'turbo_r',
            'twilight',
            'twilight_r',
            'viridis',
            'viridis_r',
            'BrBG',
            'BrBG_r',
            'PRGn',
            'PRGn_r',
            'RdBu',
            'RdBu_r',
            'RdGy',
            'RdGy_r',
            'RdYlBu',
            'RdYlBu_r'
        ]
        self.colormap_dropdown = QComboBox(self)
        self.colormap_dropdown.addItems(self.palettes)
        self.colormap_dropdown.currentIndexChanged.connect(self.update_preview)
        right_layout.addWidget(self.colormap_dropdown)

        # Image preview
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.graphics_view.viewport().setFixedSize(400, 300)  # Set viewport size
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        right_layout.addWidget(self.graphics_view)

        # Connect formula input changes to the preview update function
        self.formula_input.textChanged.connect(self.update_preview)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.on_ok_clicked)
        self.button_box.rejected.connect(self.reject)
        right_layout.addWidget(self.button_box)

    def on_ok_clicked(self):
        selected_index = self.indices_combobox.currentText()
        formula = self.formula_input.text()
        if selected_index == "Custom":
            # If formula is custom, prompt the user for a name
            self.formula_name, ok = QInputDialog.getText(self, "Formula Name", "Enter the name for your custom formula:")
            self.formula_equation = formula
            if not ok:
                return  # User canceled the input dialog
        else:
            # If formula is predefined, get the name automatically
            self.formula_name = selected_index
            self.formula_equation = formula


        self.final_result = self.compute_formula(formula)

        # Continue with any other logic you want after obtaining the formula name
        # For example, save the formula, close the dialog, etc.
        self.accept()

    def on_index_changed(self):
        selected_index = self.indices_combobox.currentText()
        if selected_index in self.indices:
            formula = self.indices[selected_index]
            self.formula_input.setText(formula)

    def add_band_to_formula(self, item):
        current_text = self.formula_input.text()
        self.formula_input.setText(current_text + item.text().split(' ')[-1][1:-1])

    def add_operator_to_formula(self, operator):
        current_text = self.formula_input.text()
        self.formula_input.setText(current_text + operator)

    def create_operator_button(self, operator):
        btn = QPushButton(operator, self)
        btn.clicked.connect(lambda: self.add_operator_to_formula(operator))
        return btn

    def check_formula(self):
        formula = self.formula_input.text()
        if self.is_formula_correct(formula):
            pixmap = QPixmap(res.find("img/check.png"))
            self.formula_indicator.setPixmap(pixmap)
        else:
            pixmap = QPixmap(res.find("img/cross.png"))
            self.formula_indicator.setPixmap(pixmap)

    def is_formula_correct(self, formula):
        # Check for balanced parentheses
        if formula.count('(') != formula.count(')'):
            return False

        # Check for incompatible operators
        operators = ['+', '-', '*', '/', '^', 'sqrt', 'sin', 'asin', 'log', 'cos', 'acos', 'tan', 'atan', 'exp']
        for op in operators:
            if formula.endswith(op) or formula.startswith(op):
                return False
            for op2 in operators:
                if op + op2 in formula or op2 + op in formula:
                    return False

        # Check for adjacent bands without an operator in between
        bands = ['B', 'G', 'PAN', 'R', 'RE', 'NIR']
        for band1 in bands:
            for band2 in bands:
                pattern = band1 + band2
                if pattern in formula:
                    return False

        return True

    def update_preview(self):
        # Get current formula
        formula = self.formula_input.text()

        # Check if formula is correct
        if not self.is_formula_correct(formula):
            # Display placeholder or clear preview
            self.graphics_scene.clear()
            return

        # Compute formula result
        result = self.compute_formula(formula)
        result = (result - np.min(result)) / (np.max(result) - np.min(result))

        # Resize the result to 400x300
        result_resized = cv2.resize(result, (400, 300))

        # Apply the colormap and visualize
        self.colormap_name = self.colormap_dropdown.currentText()
        colormap = cm.get_cmap(self.colormap_name)
        colored_result = (colormap(result_resized)[:, :, :3] * 255).astype(np.uint8)
        height, width, channel = colored_result.shape
        bytes_per_line = 3 * width
        q_image = QImage(colored_result.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Update graphics scene
        pixmap = QPixmap.fromImage(q_image)
        self.graphics_scene.clear()
        item = self.graphics_scene.addPixmap(pixmap)

        # Compute scaling factors and scale the content
        x_scale = 400 / item.boundingRect().width()
        y_scale = 300 / item.boundingRect().height()
        self.graphics_view.setTransform(QTransform().scale(x_scale, y_scale))


    def compute_formula(self, formula):
        # Sort band names by length in descending order
        sorted_band_names = sorted(self.images.keys(), key=len, reverse=True)

        # Replace band names with actual image data
        for band_name in sorted_band_names:
            pattern = r'\b' + band_name + r'\b'  # \b denotes a word boundary
            replacement = f"self.images['{band_name}']"
            formula = re.sub(pattern, replacement, formula)

        print(formula)

        # Compute the formula result
        with np.errstate(divide='ignore', invalid='ignore'):  # Handle divide by zero
            try:
                result = eval(formula)

                # Replace inf, -inf, and nan values with 0 or other desired value
                result[np.isinf(result)] = 0
                result[np.isnan(result)] = 0

                # Normalize the result to [0, 1] for visualization
                # result = (result - np.min(result)) / (np.max(result) - np.min(result))

            except Exception as e:
                print(f"Error computing formula: {e}")
                result = np.zeros_like(list(self.images.values())[0])  # Return a blank image if there's an error

        return result


class ShowComposed(QDialog):
    def __init__(self, img_paths, img_names, parent=None):
        QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'composed'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
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
        self.on_img_combo_change()

    def on_img_combo_change(self):
        i = self.comboBox_views.currentIndex()
        img_path = self.img_paths[i]
        self.viewer.setPhoto(QPixmap(img_path))


class AlignmentWindowArrow(QDialog):
    def __init__(self, ref_path, targ_path):
        super().__init__()

        # Dragging state
        self.dragging = False
        self.last_mouse_position = QPoint()

        # Load images
        self.reference_image = QImage(ref_path)
        self.to_align_image = QImage(targ_path)

        # Apply Canny edge detection
        #self.reference_image = self.apply_canny_effect(self.reference_image)
        #self.to_align_image = self.apply_canny_effect(self.to_align_image)

        # Resize 'to_align_image' to match the height of 'reference_image' while maintaining the aspect ratio
        target_width = int(self.to_align_image.width() * (self.reference_image.height() / self.to_align_image.height()))
        self.to_align_image = self.to_align_image.scaled(target_width, self.reference_image.height(),
                                                         Qt.KeepAspectRatio)

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
        self.title = QLabel('test')
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 600)

        layout.addWidget(self.title)
        layout.addWidget(self.image_label)

        # Ok and Cancel buttons
        ok_cancel_layout = QHBoxLayout()
        self.ok_button = QPushButton("Ok", self)
        self.cancel_button = QPushButton("Cancel", self)
        ok_cancel_layout.addWidget(self.ok_button)
        ok_cancel_layout.addWidget(self.cancel_button)
        layout.addLayout(ok_cancel_layout)

        # Connect the Ok and Cancel buttons to their respective functions
        self.ok_button.clicked.connect(self.save_final_image)
        self.cancel_button.clicked.connect(self.reject)

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

        # Stretch and Shrink buttons
        stretch_shrink_layout = QHBoxLayout()
        self.stretch_horizontal_button = QPushButton("Stretch Horizontally", self)
        self.shrink_horizontal_button = QPushButton("Shrink Horizontally", self)
        self.stretch_vertical_button = QPushButton("Stretch Vertically", self)
        self.shrink_vertical_button = QPushButton("Shrink Vertically", self)
        stretch_shrink_layout.addWidget(self.stretch_horizontal_button)
        stretch_shrink_layout.addWidget(self.shrink_horizontal_button)
        stretch_shrink_layout.addWidget(self.stretch_vertical_button)
        stretch_shrink_layout.addWidget(self.shrink_vertical_button)
        layout.addLayout(stretch_shrink_layout)

        # Connect the buttons to their respective functions
        self.stretch_horizontal_button.clicked.connect(self.stretch_horizontal)
        self.shrink_horizontal_button.clicked.connect(self.shrink_horizontal)
        self.stretch_vertical_button.clicked.connect(self.stretch_vertical)
        self.shrink_vertical_button.clicked.connect(self.shrink_vertical)

        # ComboBox for selecting pixel movement
        self.pixel_selector = QComboBox(self)
        self.pixel_selector.addItems(['1 pixel', '10 pixels', '20 pixels'])
        layout.addWidget(self.pixel_selector)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_position = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            # Calculate how much the mouse has moved
            dx = event.x() - self.last_mouse_position.x()
            dy = event.y() - self.last_mouse_position.y()

            # Update the offset values
            self.x_offset += dx
            self.y_offset += dy

            # Update the last mouse position
            self.last_mouse_position = event.pos()

            # Redraw the image
            self.display_images()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
    def apply_canny_effect(self, image):
        # Convert QImage to cv2 format
        cv_image = self.qimage_to_cv2(image)

        # Apply Canny edge detection
        edges = cv2.Canny(cv_image, 50, 200)  # 100 and 200 are lower and upper thresholds. Adjust if needed.

        # Convert back to QImage and return
        return self.cv2_to_qimage(edges)
    def cv2_to_qimage(self, cv_img):
        height, width = cv_img.shape
        bytes_per_line = width
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    def qimage_to_cv2(self, qimage):
        # Convert QImage to a format suitable for OpenCV
        width = qimage.width()
        height = qimage.height()

        # RGB32 format
        if qimage.format() == QImage.Format_RGB32:
            ptr = qimage.bits()
            arr = np.array(ptr).reshape(height, width, 4)  # 4 bytes per pixel
            b, g, r, a = cv2.split(arr)
            return cv2.merge([r, g, b])

        # Grayscale 8-bit format
        elif qimage.format() == QImage.Format_Grayscale8:
            ptr = qimage.bits()
            arr = np.array(ptr).reshape(height, width)  # 1 byte per pixel
            return arr

        # For other formats, use QImage's conversion capabilities before extracting the data
        else:
            qimage = qimage.convertToFormat(QImage.Format_RGB32)
            ptr = qimage.bits()
            arr = np.array(ptr).reshape(height, width, 4)
            b, g, r, a = cv2.split(arr)
            return cv2.merge([r, g, b])

    def get_translated_qimage(self):
        # Create an empty QImage with the same dimensions
        translated_image = QImage(self.to_align_image.size(), QImage.Format_ARGB32)
        translated_image.fill(Qt.transparent)

        # Use QPainter to draw the translated image
        painter = QPainter(translated_image)
        painter.drawImage(self.x_offset, self.y_offset, self.to_align_image)
        painter.end()

        return translated_image

    def save_final_image(self):
        translated_qimage = self.get_translated_qimage()
        self.cv_final_image = self.qimage_to_cv2(translated_qimage)
        # Define the ROI coordinates
        x_start = 0
        y_start = 0
        x_end = 2464
        y_end = 2056

        # Crop the image
        self.cv_final_image = self.cv_final_image[y_start:y_end, x_start:x_end]

        self.accept()

    def display_images(self):
        # Create an image for blending
        blended_image = QImage(self.reference_image.size(), QImage.Format_ARGB32)
        blended_image.fill(QColor(0, 0, 0, 0))

        # Painter for blending
        painter = QPainter(blended_image)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawImage(0, 0, self.reference_image)
        painter.setCompositionMode(QPainter.CompositionMode_Screen)
        painter.drawImage(self.x_offset, self.y_offset, self.to_align_image)

        # Limit the displayed region to the dimensions of 'reference_image'
        painter.setClipRect(0, 0, self.reference_image.width(), self.reference_image.height())

        painter.end()

        # Create QPixmap from the blended image
        pixmap = QPixmap.fromImage(blended_image)

        # Scale the QPixmap to fit the QLabel
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

        # Set the scaled pixmap to label
        self.image_label.setPixmap(pixmap)

    def stretch_horizontal(self):
        self.scale_image(1.001, 1.0)  # Increase width by 10%

    def shrink_horizontal(self):
        self.scale_image(0.999, 1.0)  # Decrease width by 10%

    def stretch_vertical(self):
        self.scale_image(1.0, 1.001)  # Increase height by 10%

    def shrink_vertical(self):
        self.scale_image(1.0, 0.999)  # Decrease height by 10%

    def scale_image(self, sx, sy):
        # Apply scaling to the to-align image
        transform = QTransform().scale(sx, sy)
        self.to_align_image = self.to_align_image.transformed(transform)
        self.display_images()

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

        self.ref_image = QImage(ref_img)
        self.target_image = QImage(target_img)

        # Resize 'to_align_image' to match the height of 'reference_image' while maintaining the aspect ratio
        target_width = int(self.target_image.width() * (self.ref_image.height() / self.target_image.height()))
        print(target_width)
        self.target_image = self.target_image.scaled(target_width, self.ref_image.height(),
                                                         Qt.KeepAspectRatio)

        self.ref_img_path = ref_img
        self.target_img_path = target_img

        self.ref_points = []
        self.target_points = []

        self.ref_view = self.create_graphics_view(self.ref_image)
        self.target_view = self.create_graphics_view(self.target_image)

        self.layout.addWidget(self.ref_view)
        self.layout.addWidget(self.target_view)

        # Controls
        control_layout = QVBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        control_layout.addWidget(self.ok_button)
        self.layout.addLayout(control_layout)

    def qimage_to_cv2(self, qimage):
        # Convert QImage to a format suitable for OpenCV
        width = qimage.width()
        height = qimage.height()

        print(qimage.format())

        # RGB32 format
        if qimage.format() == QImage.Format_RGB32:
            ptr = qimage.bits()
            arr = np.array(ptr).reshape(height, width, 4)  # 4 bytes per pixel
            b, g, r, a = cv2.split(arr)
            return cv2.merge([r, g, b])

        # Format_Grayscale16 format
        elif qimage.format() == QImage.Format_Grayscale16:
            # Given width
            width = 2751

            # Extract bytes without padding
            data = qimage.bits()
            adjusted_data_bytes = []

            for i in range(height):
                start_index = i * (width * 2 + 2)  # 2 bytes for each pixel + 2 bytes padding
                end_index = start_index + width * 2  # 2 bytes for each pixel in the current row
                adjusted_data_bytes.extend(data[start_index:end_index])

            # Convert bytes to 16-bit values
            arr_16bit = np.frombuffer(bytearray(adjusted_data_bytes), dtype=np.uint16).reshape(height, width)

            # Convert 16-bit grayscale to 8-bit grayscale for compatibility with many OpenCV functions
            arr_8bit = (arr_16bit / 256).astype(np.uint8)
            arr_8bit.shape
            return arr_8bit

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
        if len(self.ref_points) < 12 and scene == self.ref_view.scene():
            self.ref_points.append((point.x(), point.y()))
            self.add_point_marker(scene, point, len(self.ref_points))
        elif len(self.target_points) < 12 and scene == self.target_view.scene():
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
        warp = True

        # Estimate the rigid transformation
        if rigid:
            M, _ = cv2.estimateAffinePartial2D(np.array(self.target_points), np.array(self.ref_points), method=cv2.RANSAC)

            # Convert the 2x3 matrix to 3x3 to use with warpPerspective
            H = np.vstack([M, [0, 0, 1]])
        elif warp:
            H, _ = cv2.findHomography(np.array(self.target_points), np.array(self.ref_points), method=cv2.RANSAC)

        else:
            H, _ = cv2.estimateAffine2D(np.array(self.target_points), np.array(self.ref_points), method=cv2.RANSAC)

        # Warp the target image to the reference image
        target_img = self.qimage_to_cv2(self.target_image)

        # Debug: Print the shape and type of target_img
        print("Target Image Shape:", target_img.shape)
        print("Target Image Type:", type(target_img))

        if not warp:
            aligned = cv2.warpAffine(target_img, H, (self.target_image.width(), self.target_image.height()))
        else:
            aligned = cv2.warpPerspective(target_img, H, (self.target_image.width(), self.target_image.height()))

        # Define the ROI coordinates
        x_start = 0
        y_start = 0
        x_end = 2464
        y_end = 2056

        # Crop the image
        cv_final_image = aligned[y_start:y_end, x_start:x_end]

        return cv_final_image
