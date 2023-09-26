import cv2
from gui import widgets as wid
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import *
import resources as res
from scipy.optimize import basinhopping, differential_evolution, minimize
import shutil
import sys

def func(theta, ref, target):
    y_dist = int(theta[0]*100)
    x_dist = int(theta[1]*100)
    h,w = ref.shape

    print(f'theta:{theta}')

    # Slide the image
    rows, cols = target.shape
    slid_image = np.zeros_like(target)

    if x_dist > 0:
        x_end = min(cols, cols - x_dist)
        x_start = max(0, x_dist)
    else:
        x_end = min(cols, cols + x_dist)
        x_start = max(0, -x_dist)

    if y_dist > 0:
        y_end = min(rows, rows - y_dist)
        y_start = max(0, y_dist)
    else:
        y_end = min(rows, rows + y_dist)
        y_start = max(0, -y_dist)

    slid_image[y_start:y_start + y_end, x_start:x_start + x_end] = target[:y_end, :x_end]

    # create lines
    lines_ref = create_lines(ref)
    lines_target = create_lines(slid_image)

    # compute difference
    diff = cv2.subtract(lines_ref, lines_target)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))

    print(f'mse is {mse}')

    return mse


def create_lines(cv_img):
    img_gray = cv_img

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)

    # Display the image in a window named "Image"
    # cv2.imshow('Image', grad_norm)

    # Wait indefinitely for a key press (0 means wait indefinitely)
    # cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return grad_norm


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

        self.viewer = wid.PhotoViewer()
        self.verticalLayout.addWidget(self.viewer)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.comboBox_views.currentIndexChanged.connect(self.on_img_combo_change)

    def on_img_combo_change(self):
        i = self.comboBox_views.currentIndex()
        img_path = self.img_paths[i]
        self.viewer.setPhoto(QPixmap(img_path))


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
        self.scene = QGraphicsScene()
        view = wid.ClickableGraphicsView(self.scene)
        pixmap = QPixmap.fromImage(img)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap.rect())
        view.pointClicked.connect(lambda point, s=self.scene: self.on_image_click(point, s))
        return view

    def load_points(self, point_list):
        self.ref_points = point_list
        for point in point_list:
            self.add_point_marker(self.scene, point, len(self.ref_points))

    def on_image_click(self, point, scene):
        # point is now directly in scene's coordinates
        if len(self.ref_points) < 6 and scene == self.ref_view.scene():
            self.ref_points.append((point.x(), point.y()))
            self.add_point_marker(scene, point, len(self.ref_points))
        elif len(self.target_points) < 6 and scene == self.target_view.scene():
            self.target_points.append((point.x(), point.y()))
            self.add_point_marker(scene, point, len(self.target_points))

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
        # self.palettes = sorted(plt.colormaps())
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
        self.palette_combobox.addItems(self.palettes)
        self.palette_combobox.currentIndexChanged.connect(self.update_display)
        self.layout.addWidget(self.palette_combobox)

        self.imageviewer = wid.PhotoViewer(self)
        self.layout.addWidget(self.imageviewer)

        self.align_button = QPushButton("Align This Shot")
        self.align_button.clicked.connect(self.align_images_manual)
        self.layout.addWidget(self.align_button)

        self.see_composed_button = QPushButton("See Composed Images", self)
        self.see_composed_button.setEnabled(False)
        self.see_composed_button.clicked.connect(self.show_composed_shots)

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

        if os.path.isdir(os.path.join(self.base_dir, self.selected_shot)):
            self.see_composed_button.setEnabled(True)
            self.selected_compo = item.text()
            self.show_composed_shots()


        else:
            self.see_composed_button.setEnabled(False)
            self.update_display()

    def show_composed_shots(self):
        # Load individual channel images
        red_path = os.path.join(self.base_dir, self.selected_compo, 'aligned_3.tif')
        green_path = os.path.join(self.base_dir, self.selected_compo, 'aligned_2.tif')
        blue_path = os.path.join(self.base_dir, self.selected_compo, 'aligned_1.tif')
        rededge_path = os.path.join(self.base_dir, self.selected_compo, 'aligned_5.tif')
        nir_path = os.path.join(self.base_dir, self.selected_compo, 'aligned_4.tif')

        red_channel_img = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)
        re_channel_img = cv2.imread(rededge_path, cv2.IMREAD_GRAYSCALE)
        nir_channel_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        green_channel_img = cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)
        blue_channel_img = cv2.imread(blue_path, cv2.IMREAD_GRAYSCALE)

        #RGB Image
        rgb_image = cv2.merge((red_channel_img, green_channel_img, blue_channel_img))
        out_rgb_path = os.path.join(self.base_dir, self.selected_compo, 'rgb.png')
        cv2.imwrite(out_rgb_path, rgb_image)

        # Rededge G B image
        regb_image = cv2.merge((re_channel_img, green_channel_img, blue_channel_img))
        out_regb_path = os.path.join(self.base_dir, self.selected_compo, 'regb.png')
        cv2.imwrite(out_regb_path, regb_image)

        # NIR R G image
        cir_image = cv2.merge((nir_channel_img, red_channel_img, blue_channel_img))
        out_cir_path = os.path.join(self.base_dir, self.selected_compo, 'cir.png')
        cv2.imwrite(out_cir_path, cir_image)

        # img_names = ['NDVI', 'GNDVI', 'EVI', 'RENDVI', 'SR']
        img_names = ['RGB', 'REGB', 'CIR']
        img_paths = [out_rgb_path, out_regb_path, out_cir_path]

        dialog = ShowComposed(img_paths, img_names)
        if dialog.exec_():
            pass

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


    def align_images_manual(self):
        ref_img_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_1.tif")  # Using 1st band as reference

        shot_name = f"ALIGNED_IMG_{self.selected_shot}"
        aligned_folder_path = os.path.join(self.base_dir, shot_name)
        if not os.path.exists(aligned_folder_path):
            os.mkdir(aligned_folder_path)

        # copy reference
        dst_ref = os.path.join(aligned_folder_path, f"aligned_1.tif")
        shutil.copyfile(ref_img_path, dst_ref)

        for i in range(2, 6):  # Start from 2 since 1 is the reference
            target_img_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_{i}.tif")
            if i == 2:
                alignment_window = AlignmentWindow(ref_img_path, target_img_path)
                if alignment_window.exec_() == QDialog.Accepted:
                    aligned_image = alignment_window.get_aligned_image()
                    ref_points = alignment_window.ref_points
            else:
                alignment_window = AlignmentWindow(ref_img_path, target_img_path)
                alignment_window.load_points(ref_points)
                if alignment_window.exec_() == QDialog.Accepted:
                    ref_points = alignment_window.ref_points
                    aligned_image = alignment_window.get_aligned_image()

            # Save the aligned image
            aligned_filename = os.path.join(aligned_folder_path,
                                            "aligned_{}.tif".format(i))
            cv2.imwrite(aligned_filename, aligned_image)

        # add new folder element in the listview
        folder_img = res.find('img/folder.png')
        item = QListWidgetItem(QIcon(folder_img), shot_name)
        self.shot_list.addItem(item)

    # OLD METHODS
    def align_images_orb(self, ref, target):
        # Convert images to 8-bit for SIFT
        ref = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        target = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        orb_detector = cv2.ORB_create(5000)

        kp1, d1 = orb_detector.detectAndCompute(ref, None)
        kp2, d2 = orb_detector.detectAndCompute(target, None)

        # Match features between the two images.
        # We create a Brute Force matcher with
        # Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = list(matcher.match(d1, d2))

        # Sort matches on the basis of their Hamming distance.
        matches.sort(key=lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches) * 0.9)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

        # Use this matrix to transform the
        # colored image wrt the reference image.
        height, width = target.shape
        aligned = cv2.warpPerspective(target,
                                              homography, (width, height))

        return aligned


    def align_current_shot(self):
        ref_image_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_1.tif")  # Using 1st band as reference
        ref_image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)

        for i in range(2, 5):  # Start from 2 since 1 is the reference
            target_image_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_{i}.tif")
            target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)

            self.align_images_lines(ref_image, target_image, target_image_path)


    def align_images_lines(self, ref, target, dest_path):
        res = differential_evolution(func, ((-1, 1), (-1, 1)), maxiter=50, args=(ref,target,))

        print(res)
        print(res.x)

        y_dist = int(res.x[0])
        x_dist = int(res.x[1])

        # Slide the image
        rows, cols = target.shape
        slid_image = np.zeros_like(target)

        if x_dist > 0:
            x_end = min(cols, cols - x_dist)
            x_start = max(0, x_dist)
        else:
            x_end = min(cols, cols + x_dist)
            x_start = max(0, -x_dist)

        if y_dist > 0:
            y_end = min(rows, rows - y_dist)
            y_start = max(0, y_dist)
        else:
            y_end = min(rows, rows + y_dist)
            y_start = max(0, -y_dist)

        slid_image[y_start:y_start + y_end, x_start:x_start + x_end] = target[:y_end, :x_end]

        save_path = os.path.splitext(dest_path)[0] + "_aligned.tif"
        cv2.imwrite(save_path, slid_image)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.showMaximized()
    sys.exit(app.exec())