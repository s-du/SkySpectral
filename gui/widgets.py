# standard libraries
import logging
import numpy as np
import os
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
import sys
import traceback


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# basic logger functionality
log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)


def show_exception_box(log_msg):
    """Checks if a QApplication instance is available and shows a messagebox with the exception message.
    If unavailable (non-console application), log an additional notice.
    """
    if QApplication.instance() is not None:
        errorbox = QMessageBox()
        errorbox.setText("Oops. An unexpected error occured:\n{0}".format(log_msg))
        errorbox.exec_()
    else:
        log.debug("No QApplication instance available.")


class UncaughtHook(QObject):
    _exception_caught = Signal(object)

    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        # this registers the exception_hook() function as hook with the Python interpreter
        sys.excepthook = self.exception_hook

        # connect signal to execute the message box function always on main thread
        self._exception_caught.connect(show_exception_box)

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """Function handling uncaught exceptions.
        It is triggered each time an uncaught exception occurs.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Uncaught exception:\n {0}".format(log_msg), exc_info=exc_info)

            # trigger message box show
            self._exception_caught.emit(log_msg)


# create a global instance of our class to register the hook
qt_exception_hook = UncaughtHook()


class UiLoader(QUiLoader):
    """
    Subclass :class:`~PySide.QtUiTools.QUiLoader` to create the user interface
    in a base instance.

    Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
    create a new instance of the top-level widget, but creates the user
    interface in an existing instance of the top-level class.

    This mimics the behaviour of :func:`PyQt4.uic.loadUi`.
    """

    def __init__(self, baseinstance, customWidgets=None):
        """
        Create a loader for the given ``baseinstance``.

        The user interface is created in ``baseinstance``, which must be an
        instance of the top-level class in the user interface to load, or a
        subclass thereof.

        ``customWidgets`` is a dictionary mapping from class name to class object
        for widgets that you've promoted in the Qt Designer interface. Usually,
        this should be done by calling registerCustomWidget on the QUiLoader, but
        with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

        ``parent`` is the parent object of this loader.
        """

        QUiLoader.__init__(self, baseinstance)
        self.baseinstance = baseinstance
        self.customWidgets = customWidgets

    def createWidget(self, class_name, parent=None, name=''):
        """
        Function that is called for each widget defined in ui file,
        overridden here to populate baseinstance instead.
        """

        if parent is None and self.baseinstance:
            # supposed to create the top-level widget, return the base instance
            # instead
            return self.baseinstance

        else:
            if class_name in self.availableWidgets():
                # create a new widget for child widgets
                widget = QUiLoader.createWidget(self, class_name, parent, name)

            else:
                # if not in the list of availableWidgets, must be a custom widget
                # this will raise KeyError if the user has not supplied the
                # relevant class_name in the dictionary, or TypeError, if
                # customWidgets is None
                try:
                    widget = self.customWidgets[class_name](parent)

                except (TypeError, KeyError) as e:
                    raise Exception(
                        'No custom widget ' + class_name + ' found in customWidgets param of UiLoader __init__.')

            if self.baseinstance:
                # set an attribute for the new child widget on the base
                # instance, just like PyQt4.uic.loadUi does.
                setattr(self.baseinstance, name, widget)

                # this outputs the various widget names, e.g.
                # sampleGraphicsView, dockWidget, samplesTableView etc.
                # print(name)

            return widget


def loadUi(uifile, baseinstance=None, customWidgets=None,
           workingDirectory=None):
    """
    Dynamically load a user interface from the given ``uifile``.

    ``uifile`` is a string containing a file name of the UI file to load.

    If ``baseinstance`` is ``None``, the a new instance of the top-level widget
    will be created.  Otherwise, the user interface is created within the given
    ``baseinstance``.  In this case ``baseinstance`` must be an instance of the
    top-level widget class in the UI file to load, or a subclass thereof.  In
    other words, if you've created a ``QMainWindow`` interface in the designer,
    ``baseinstance`` must be a ``QMainWindow`` or a subclass thereof, too.  You
    cannot load a ``QMainWindow`` UI file with a plain
    :class:`~PySide.QtGui.QWidget` as ``baseinstance``.

    ``customWidgets`` is a dictionary mapping from class name to class object
    for widgets that you've promoted in the Qt Designer interface. Usually,
    this should be done by calling registerCustomWidget on the QUiLoader, but
    with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

    :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on the
    created user interface, so you can implemented your slots according to its
    conventions in your widget class.

    Return ``baseinstance``, if ``baseinstance`` is not ``None``.  Otherwise
    return the newly created instance of the user interface.
    """

    loader = UiLoader(baseinstance, customWidgets)

    if workingDirectory is not None:
        loader.setWorkingDirectory(workingDirectory)

    widget = loader.load(uifile)
    QMetaObject.connectSlotsByName(widget)
    return widget

def create_line_iterator(P1, P2, img):
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


class DoubleSlider(QSlider):
    doubleValueChanged = Signal(int, int)

    def __init__(self, *args, **kwargs):
        super(DoubleSlider, self).__init__(*args, **kwargs)
        self._low = self.minimum()
        self._high = self.maximum()
        self.pressed_control = QStyle.SC_None
        self.setOrientation(Qt.Horizontal)

    def low(self):
        return self._low

    def setLow(self, low):
        if low != self._low:
            self._low = low
            self.doubleValueChanged.emit(self._low, self._high)
            self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        if high != self._high:
            self._high = high
            self.doubleValueChanged.emit(self._low, self._high)
            self.update()

    def mousePressEvent(self, event):
        event.accept()

        # Determine which thumb is closer to the mouse press position
        distance_to_low = abs(event.x() - self._pixelPos(self._low))
        distance_to_high = abs(event.x() - self._pixelPos(self._high))

        if distance_to_low < distance_to_high:
            self.pressed_control = "low"
            self._low = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())
        else:
            self.pressed_control = "high"
            self._high = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())

        self.update()

    def mouseMoveEvent(self, event):
        event.accept()
        new_position = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())

        if self.pressed_control == "low":
            if new_position <= self._high:
                self.setLow(new_position)
        elif self.pressed_control == "high":
            if new_position >= self._low:
                self.setHigh(new_position)

        self.update()

    def mouseReleaseEvent(self, event):
        self.pressed_control = None
        self.update()

    def _pixelPos(self, value):
        """ Convert slider value to pixel position """
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return self.style().sliderPositionFromValue(self.minimum(), self.maximum(), value, self.width())

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw the background
        background = QColor(0, 0, 0, 50)
        painter.setPen(Qt.NoPen)
        painter.setBrush(background)
        painter.drawRect(0, self.height() / 4, self.width(), self.height() / 2)

        # Draw the range
        groove_color = self.palette().color(self.backgroundRole()).darker(110)
        painter.setBrush(groove_color)
        slider_range = QStyleOptionSlider()
        self.initStyleOption(slider_range)
        width = self.style().pixelMetric(QStyle.PM_SliderLength, slider_range, self)
        min_loc = self.style().sliderPositionFromValue(self.minimum(), self.maximum(), self.low(), self.width())
        max_loc = self.style().sliderPositionFromValue(self.minimum(), self.maximum(), self.high(), self.width())
        painter.drawRect(min_loc, self.height() / 4, max_loc - min_loc, self.height() / 2)

        # Draw the handles
        handle = QStyleOptionSlider()
        self.initStyleOption(handle)
        handle.sliderPosition = self.low()
        handle.subControls = QStyle.SC_SliderHandle
        self.style().drawComplexControl(QStyle.CC_Slider, handle, painter, self)

        handle.sliderPosition = self.high()
        self.style().drawComplexControl(QStyle.CC_Slider, handle, painter, self)


class MagnifyingGlass(QGraphicsEllipseItem):
    def __init__(self, size=200, parent=None):
        self._size = size
        super().__init__(-self._size/2, -self._size/2, self._size, self._size, parent)
        self.setBrush(Qt.transparent)
        self.setPen(Qt.NoPen)
        self.pixmap_item = QGraphicsPixmapItem(self)
        self.pixmap_item.setPos(-self._size/2, -self._size/2)
        self.setZValue(1)

    def set_pixmap(self, pixmap):
        # Create an elliptical mask for the pixmap
        mask_image = QImage(pixmap.size(), QImage.Format_Alpha8)
        mask_image.fill(Qt.transparent)

        painter = QPainter(mask_image)
        painter.setBrush(Qt.white)
        painter.drawEllipse(mask_image.rect())
        painter.end()

        # Convert QImage to QBitmap for PySide6
        mask_bitmap = QBitmap.fromImage(mask_image)

        pixmap.setMask(mask_bitmap)

        # Set the masked pixmap to the pixmap item
        self.pixmap_item.setPixmap(pixmap)

class ClickableGraphicsView(QGraphicsView):
    # Create a custom signal that will emit the clicked point
    pointClicked = Signal(QPointF)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.ArrowCursor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setFrameShape(QFrame.NoFrame)

        self.magnifying_glass_size = 400  # Adjust this value to change the size
        self.magnifying_glass = MagnifyingGlass(self.magnifying_glass_size)
        self.scene().addItem(self.magnifying_glass)
    def mouseMoveEvent(self, event):
        # Position in scene coordinates
        scene_pos = self.mapToScene(event.pos())

        # Temporarily hide the magnifying glass to avoid rendering it
        self.magnifying_glass.hide()

        # Render the scene into an image
        scene_image = QImage(self.scene().sceneRect().size().toSize(), QImage.Format_ARGB32)
        scene_image.fill(Qt.transparent)
        painter = QPainter(scene_image)
        self.scene().render(painter)
        painter.end()

        # Adjust these values for desired magnification
        magnify_factor = 3

        # Calculate the dimensions of the sub-pixmap to grab
        grab_width = self.magnifying_glass_size // magnify_factor
        grab_height = self.magnifying_glass_size // magnify_factor

        # Calculate the top-left corner of the sub-pixmap to grab, such that the cursor is centered
        grab_x = scene_pos.x() - grab_width / 2
        grab_y = scene_pos.y() - grab_height / 2

        # Extract the portion of the rendered scene around the cursor
        sub_image = scene_image.copy(grab_x, grab_y, grab_width, grab_height)

        # Convert QImage to QPixmap and scale it to achieve magnification
        magnified_pixmap = QPixmap.fromImage(sub_image).scaled(self.magnifying_glass_size, self.magnifying_glass_size,
                                                               Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Update the magnifying glass
        self.magnifying_glass.setPos(scene_pos)
        self.magnifying_glass.set_pixmap(magnified_pixmap)
        self.magnifying_glass.show()

    def mousePressEvent(self, event):
        # Emit the clicked point in scene coordinates
        self.pointClicked.emit(self.mapToScene(event.pos()))
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        """
        if event.angleDelta().y() > 0:
            self.resetTransform()
            self.scale(1.5, 1.5)
        """
        pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

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
                line_values = create_line_iterator(p1, p2, self.height_values)

                self.line_values_final = line_values[:,2]

                # emit signal (end of measure)
                self.endDrawing_line_meas.emit()
                print('Line meas. added')

            self.origin = QPoint()
            self._current_line_item = None
            self.toggleDragMode()

        super(PhotoViewer, self).mouseReleaseEvent(event)