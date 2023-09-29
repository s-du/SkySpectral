import cv2
from gui import dialogs as dia
from gui import widgets as wid
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import resources as res
from scipy.optimize import basinhopping, differential_evolution, minimize
import shutil
import sys


class MultiSpectralIndice:
    def __init__(self, name):
        self.array = []
        self.equation = ''
        self.name = name
        self.bounds = []
        self.img_path = ''
        self.palette = ''


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Micasense Rededge P Image Processor")

        basepath = os.path.dirname(__file__)
        basename = 'main'
        uifile = os.path.join(basepath, 'gui/ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        self.setWindowTitle('Process Micasense Rededge P images')

        # UI Elements
        self.shot_list.setViewMode(QListWidget.IconMode)
        self.shot_list.setLayoutDirection(Qt.LeftToRight)

        self.shot_list.setIconSize(QSize(64, 64))  # Set the desired icon size
        self.selected_shot = ''

        # Band ComboBox
        self.band_combobox.addItems(['blue', 'green', 'red', 'NIR', 'red edge', 'panchromatic'])

        # Palette ComboBox
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

        # image viewer
        self.imageviewer = wid.PhotoViewer(self)
        self.verticalLayout_2.addWidget(self.imageviewer)

        # Variables for our images
        self.base_dir = ""
        self.shots = []

        # add icons
        self.add_icon(res.find('img/folder.png'), self.actionLoad)
        self.add_icon(res.find('img/point.png'), self.actionAlignPoints)
        self.add_icon(res.find('img/arrow.png'), self.actionAlignArrows)
        self.add_icon(res.find('img/profile.png'), self.actionShowCompo)
        self.add_icon(res.find('img/factory.png'), self.actionPrepareAgisoft)
        self.add_icon(res.find('img/math.png'), self.actionAddTransform)

        # slots
        self.create_connections()

    def create_connections(self):
        self.actionLoad.triggered.connect(self.load_images)
        self.actionAlignPoints.triggered.connect(self.align_images_manual)
        self.actionAlignArrows.triggered.connect(self.align_images_arrows)
        self.actionShowCompo.triggered.connect(self.show_composed_shots)
        self.actionPrepareAgisoft.triggered.connect(self.prepare_agisoft)
        self.actionAddTransform.triggered.connect(self.raster_transform)

        self.band_combobox.currentIndexChanged.connect(self.update_display)
        self.palette_combobox.currentIndexChanged.connect(self.update_display)
        self.shot_list.itemClicked.connect(self.shot_selected)

    def add_icon(self, img_source, pushButton_object):
        pushButton_object.setIcon(QIcon(img_source))

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

        # enable actions
        self.actionAlignPoints.setEnabled(True)
        self.actionAlignArrows.setEnabled(True)
        self.actionPrepareAgisoft.setEnabled(True)

        self.band_combobox.setEnabled(True)
        self.palette_combobox.setEnabled(True)

    def prepare_agisoft(self):
        # Create the 'for Agisoft' folder
        agisoft_folder = os.path.join(self.base_dir, 'for_Agisoft')
        os.makedirs(agisoft_folder, exist_ok=True)

        # Band mapping
        band_mapping = {
            '1': 'Blue',
            '2': 'Green',
            '3': 'Red',
            '4': 'NIR',
            '5': 'RedEdge',
            '6': 'A-Panchromatic'
        }

        # For each image in the folder
        for filename in os.listdir(self.base_dir):
            if filename.endswith('.tif') and '_' in filename:
                band_number = filename.split('_')[-1].split('.')[0]
                band_name = band_mapping.get(band_number)

                if band_name:
                    # Create the band subfolder inside 'for Agisoft'
                    band_folder = os.path.join(agisoft_folder, band_name)
                    os.makedirs(band_folder, exist_ok=True)

                    # Move the image to its respective band folder
                    src_path = os.path.join(self.base_dir, filename)
                    dst_path = os.path.join(band_folder, filename)
                    shutil.copy(src_path, dst_path)

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
            self.actionShowCompo.setEnabled(True)
            self.selected_compo = item.text()
        else:
            self.actionShowCompo.setEnabled(False)
            self.update_display()

    def create_main_indices(self, images_path):
        """
        Depreciated
        """
        indices = []
        red = cv2.imread(images_path[0], cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        red_edge = cv2.imread(images_path[1], cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        nir = cv2.imread(images_path[2], cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        green = cv2.imread(images_path[3], cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        blue = cv2.imread(images_path[4], cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

        # Calculate NDVI
        ndvi = MultiSpectralIndice('NDVI (Normalized Difference Vegetation Index)')
        ndvi.array = (nir - red) / (nir + red + 1e-10)  # added small value to prevent division by zero
        ndvi.bounds = [-1, 1]

        # Calculate SR
        sr = MultiSpectralIndice('SR (Simple ratio)')
        sr.array = nir / (red + 1e-10)
        sr.bounds = [0, 1]

        # Calculate MSI (Moisture stress index)
        msi = MultiSpectralIndice('MSI (Moisture stress index)')
        msi.array = nir / (red_edge + 1e-10)
        msi.bounds = [0, 1]

        # Calculate NDWI (Normalized Difference Water Index)
        ndwi = MultiSpectralIndice('NDWI (Normalized Difference Water Index)')
        ndwi.array = (green - nir) / (green + nir + 1e-10)
        ndwi.bounds = [-1, 1]

        indices.append(ndvi)
        indices.append(sr)
        indices.append(msi)
        indices.append(ndwi)

        return indices

    def raster_transform(self):
        self.get_channels_paths()
        dialog = dia.RasterTransformDialog(self.images)
        if dialog.exec_():
            # create new indice from user choices
            indice = MultiSpectralIndice(dialog.formula_name)
            indice.equation = dialog.formula_equation
            indice.array = dialog.final_result
            indice.bounds = [np.amin(indice.array), np.amax(indice.array)]
            indice.palette = dialog.colormap_name

            # output an image
            sub_compo = os.path.join(self.base_dir, self.selected_compo, 'composed')

            plt.imshow(indice.array, cmap=indice.palette, vmin=indice.bounds[0],
                       vmax=indice.bounds[1])  # set color limits to -1 and 1
            plt.colorbar()
            plt.title(indice.name + ' (' + indice.equation + ')')
            plt.axis('off')
            indice.img_path = os.path.join(sub_compo, f'{indice.name}.png')
            plt.savefig(indice.img_path, dpi=300, bbox_inches='tight')
            plt.clf()



    def create_compo_shots(self):
        self.get_channels_paths()
        sub_compo = os.path.join(self.base_dir, self.selected_compo, 'composed')

        if not os.path.exists(sub_compo):
            os.makedirs(sub_compo)

        # Convert the float images in the range [0, 1] back to uint8 in the range [0, 255]
        converted_images = {k: (v * 255).astype(np.uint8) for k, v in self.images.items()}

        # PANSHARPENING RGB


        # RGB Image
        rgb_image = cv2.merge((converted_images['B'], converted_images['G'], converted_images['R']))
        out_rgb_path = os.path.join(sub_compo, 'rgb.png')
        cv2.imwrite(out_rgb_path, rgb_image)

        # Rededge G B image
        regb_image = cv2.merge((converted_images['B'], converted_images['G'], converted_images['RE']))
        out_regb_path = os.path.join(sub_compo, 'regb.png')
        cv2.imwrite(out_regb_path, regb_image)

        # NIR R G image
        cir_image = cv2.merge((converted_images['G'], converted_images['R'], converted_images['NIR']))
        out_cir_path = os.path.join(sub_compo, 'cir.png')
        cv2.imwrite(out_cir_path, cir_image)

    def get_channels_paths(self):
        # Load individual channel images
        images_paths = [
            os.path.join(self.base_dir, self.selected_compo, 'aligned_1.tif'),
            os.path.join(self.base_dir, self.selected_compo, 'aligned_2.tif'),
            os.path.join(self.base_dir, self.selected_compo, 'aligned_3.tif'),
            os.path.join(self.base_dir, self.selected_compo, 'aligned_4.tif'),
            os.path.join(self.base_dir, self.selected_compo, 'aligned_5.tif'),
        ]
        self.images = {name: cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
                       for name, path in zip(['B', 'G', 'R', 'NIR', 'RE'], images_paths)}

    def show_composed_shots(self):
        img_files = os.listdir(os.path.join(self.base_dir, self.selected_compo, 'composed'))
        img_paths = []
        img_names = []
        for img_file in img_files:
            img_paths.append(os.path.join(self.base_dir, self.selected_compo, 'composed', img_file))
            img_names.append(img_file[:-4])

        # indices = self.create_main_indices(images)
        dialog = dia.ShowComposed(img_paths, img_names)
        if dialog.exec_():
            pass

        """
        #  Create indices images
        for indice in indices:
            # Display the NDVI image
            plt.imshow(indice.array, cmap='Spectral', vmin=indice.bounds[0], vmax=indice.bounds[1])  # set color limits to -1 and 1
            plt.colorbar()
            plt.title(indice.name)
            plt.axis('off')
            indice.img_path = os.path.join(self.base_dir, self.selected_compo, f'{indice.name}.png')
            plt.savefig(indice.img_path, dpi=300, bbox_inches='tight')
            plt.clf()

        # img_names = ['NDVI', 'GNDVI', 'EVI', 'RENDVI', 'SR']

        for indice in indices:
            img_names.append(indice.name)
            img_paths.append(indice.img_path)
        """

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

    def align_images_arrows(self):
        ref_img_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_6.tif")  # Using PAN band as reference

        aligned_folder_path = os.path.join(self.base_dir, self.selected_shot)
        if not os.path.exists(aligned_folder_path):
            os.mkdir(aligned_folder_path)

        dst_ref = os.path.join(aligned_folder_path, f"aligned_6.tif")
        shutil.copyfile(ref_img_path, dst_ref)

        for i in range(1, 6):  # Start from 2 since 1 is the reference
            target_img_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_{i}.tif")
            alignment_window = dia.AlignmentWindowArrow(ref_img_path, target_img_path)
            if alignment_window.exec_() == QDialog.Accepted:
                aligned_image = alignment_window.cv_final_image

            # Save the aligned image
            aligned_filename = os.path.join(aligned_folder_path,
                                            "aligned_{}.tif".format(i))
            cv2.imwrite(aligned_filename, aligned_image)

        self.actionShowCompo.setEnabled(True)
        self.actionAddTransform.setEnabled(True)
        self.selected_compo = self.selected_shot
        self.create_compo_shots()
        """
        # add new folder element in the listview
        folder_img = res.find('img/folder.png')
        item = QListWidgetItem(QIcon(folder_img), shot_name)
        self.shot_list.addItem(item)
        """

    def align_images_manual(self):
        ref_img_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_6.tif")  # Using pan band as reference

        shot_name = f"ALIGNED_IMG_{self.selected_shot}"
        aligned_folder_path = os.path.join(self.base_dir, shot_name)
        if not os.path.exists(aligned_folder_path):
            os.mkdir(aligned_folder_path)

        # copy reference
        dst_ref = os.path.join(aligned_folder_path, f"aligned_6.tif")
        shutil.copyfile(ref_img_path, dst_ref)

        for i in range(1, 6):  # Start from 2 since 1 is the reference
            target_img_path = os.path.join(self.base_dir, f"IMG_{self.selected_shot}_{i}.tif")
            if i == 1:
                alignment_window = dia.AlignmentWindow(ref_img_path, target_img_path)
                if alignment_window.exec_() == QDialog.Accepted:
                    aligned_image = alignment_window.get_aligned_image()
                    ref_points = alignment_window.ref_points
            else:
                alignment_window = dia.AlignmentWindow(ref_img_path, target_img_path)
                alignment_window.load_points(ref_points)
                if alignment_window.exec_() == QDialog.Accepted:
                    ref_points = alignment_window.ref_points
                    aligned_image = alignment_window.get_aligned_image()

            # Save the aligned image
            aligned_filename = os.path.join(aligned_folder_path,
                                            "aligned_{}.tif".format(i))
            cv2.imwrite(aligned_filename, aligned_image)

        self.actionShowCompo.setEnabled(True)
        self.actionAddTransform.setEnabled(True)
        self.selected_compo = self.selected_shot
        self.create_compo_shots()

    # OLD METHODS
    """
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
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.showMaximized()
    sys.exit(app.exec())
