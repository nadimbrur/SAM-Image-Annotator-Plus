import sys
import os
import json
import cv2
import numpy as np
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QHBoxLayout, QWidget, QPushButton, QComboBox, 
                           QFileDialog, QMessageBox, QDialog, QScrollArea)
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QWheelEvent, QMouseEvent, QCursor
from PyQt6.QtCore import Qt, QPoint
from segment_anything import sam_model_registry, SamPredictor

class ZoomableQLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.base_pixmap = None
        # self.setMinimumSize(800, 600)
 
        self.pan_start = None
        self.scroll_area = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_scroll_area(self, scroll_area):
        self.scroll_area = scroll_area

    def wheelEvent(self, event: QWheelEvent):
        if self.base_pixmap:
            # Get the position of the cursor relative to the image
            mouse_pos = event.position()
            widget_pos = self.mapFromGlobal(self.mapToGlobal(mouse_pos.toPoint()))
            
            # Calculate relative position within the image
            rel_x = widget_pos.x() / self.width()
            rel_y = widget_pos.y() / self.height()

            # Calculate zoom
            steps = event.angleDelta().y() / 120
            zoom_change = 1.1 ** steps
            new_zoom = self.zoom_factor * zoom_change
            
            # Limit zoom range
            if 0.1 <= new_zoom <= 5.0:
                old_zoom = self.zoom_factor
                self.zoom_factor = new_zoom
                self.update_zoom()

                # Adjust scroll position to keep mouse point fixed
                if self.scroll_area:
                    # Calculate new scroll position
                    new_width = self.pixmap().width()
                    new_height = self.pixmap().height()
                    
                    # Calculate new scroll positions
                    h_scroll = self.scroll_area.horizontalScrollBar()
                    v_scroll = self.scroll_area.verticalScrollBar()
                    
                    # Adjust scroll position to keep the mouse point fixed
                    h_scroll.setValue(int(rel_x * new_width - widget_pos.x() + h_scroll.value() * (new_zoom/old_zoom)))
                    v_scroll.setValue(int(rel_y * new_height - widget_pos.y() + v_scroll.value() * (new_zoom/old_zoom)))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_start and self.scroll_area:
            delta = event.pos() - self.pan_start
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.pan_start = event.pos()
            event.accept()
        elif event.buttons() == Qt.MouseButton.MiddleButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_base_pixmap(self, pixmap: QPixmap):
        self.base_pixmap = pixmap
        self.update_zoom()

    def update_zoom(self):
        if self.base_pixmap:
            # Calculate scaled size
            scaled_size = self.base_pixmap.size() * self.zoom_factor
            scaled_pixmap = self.base_pixmap.scaled(
                scaled_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)

    def get_image_coordinates(self, event_pos):
        if not self.base_pixmap or not self.pixmap():
            return None

        # Get the current widget size and pixmap size
        widget_size = self.size()
        pixmap_size = self.pixmap().size()
        
        # Calculate the actual image area within the widget (accounting for center alignment)
        image_rect = self.pixmap().rect()
        image_rect.moveCenter(self.rect().center())
        
        # Get the position relative to the image area
        pos_in_widget = event_pos - image_rect.topLeft()
        
        # Convert the position to original image coordinates
        original_x = int(pos_in_widget.x() / self.zoom_factor)
        original_y = int(pos_in_widget.y() / self.zoom_factor)
        
        # Ensure coordinates are within bounds
        original_x = max(0, min(original_x, self.base_pixmap.width() - 1))
        original_y = max(0, min(original_y, self.base_pixmap.height() - 1))
        
        return QPoint(original_x, original_y)

class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        layout = QVBoxLayout(self)
        
        shortcuts = [
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Ctrl+S", "Save Annotations"),
            ("E", "Toggle Erase Mode"),
            ("C", "Clear Points"),
            ("Right Arrow", "Next Image"),
            ("Left Arrow", "Previous Image"),
            ("Enter", "Save Object"),
            ("O", "Open File"),
            ("H", "Show this Help"),
        ]
        
        for key, description in shortcuts:
            layout.addWidget(QLabel(f"{key}: {description}"))
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM-2 Image Annotator")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.current_image = None
        self.image_path = None
        self.image_list = []
        self.current_image_index = -1
        self.points = []
        self.labels = []
        self.annotations = {}
        self.sam_predictor = None
        self.current_mask = None
        
        # Store finished annotations for current image
        self.finished_masks = []  # List of (mask, color) tuples
        
        # Load labels from file
        self.label_options = self.load_labels()
        
        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Erase mode
        self.erase_mode = False

        self.init_ui()
        self.init_sam_model()
        self.setup_shortcuts()

    def load_labels(self):
        labels = []
        try:
            with open('food_labels.txt', 'r') as f:
                for line in f:
                    # Skip empty lines and comments
                    line = line.strip()
                    if line and not line.startswith('#'):
                        labels.append(line)
        except FileNotFoundError:
            # Fallback to default labels if file not found
            labels = ["biryani", "pizza", "burger"]
            
        return labels

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create zoomable image label
        # self.image_label = ZoomableQLabel()
        # self.image_label.mousePressEvent = self.on_image_click
        # self.image_label.set_scroll_area(self.scroll_area)
        # self.scroll_area.setWidget(self.image_label)

        # self.image_label = QLabel()
        self.image_label = ZoomableQLabel()
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)





        # Add info label to status bar
        self.statusBar().showMessage("Middle mouse button or middle click to pan")

        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # Add buttons
        self.open_btn = QPushButton("Open Image/Directory (O)")
        self.open_btn.clicked.connect(self.open_file)

        self.label_combo = QComboBox()
        self.label_combo.addItems(self.label_options)

        self.save_object_btn = QPushButton("Save Object (Enter)")
        self.save_object_btn.clicked.connect(self.save_current_object)

        self.next_image_btn = QPushButton("Next Image (→)")
        self.next_image_btn.clicked.connect(self.next_image)

        self.prev_image_btn = QPushButton("Previous Image (←)")
        self.prev_image_btn.clicked.connect(self.prev_image)

        self.clear_points_btn = QPushButton("Clear Points (C)")
        self.clear_points_btn.clicked.connect(self.clear_points)

        self.erase_btn = QPushButton("Toggle Erase Mode (E)")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(self.toggle_erase_mode)

        self.shortcuts_btn = QPushButton("Show Shortcuts (H)")
        self.shortcuts_btn.clicked.connect(self.show_shortcuts)

        # Add widgets to control panel
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(self.label_combo)
        control_layout.addWidget(self.save_object_btn)
        control_layout.addWidget(self.next_image_btn)
        control_layout.addWidget(self.prev_image_btn)
        control_layout.addWidget(self.clear_points_btn)
        control_layout.addWidget(self.erase_btn)
        control_layout.addWidget(self.shortcuts_btn)
        control_layout.addStretch()

        # Add to main layout
        layout.addWidget(self.scroll_area)
        layout.addWidget(control_panel)

    def setup_shortcuts(self):
        # Save annotations
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_annotations)
        
        # Undo/Redo
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.redo)
        
        # Navigation
        self.next_shortcut = QShortcut(QKeySequence("Right"), self)
        self.next_shortcut.activated.connect(self.next_image)
        
        self.prev_shortcut = QShortcut(QKeySequence("Left"), self)
        self.prev_shortcut.activated.connect(self.prev_image)
        
        # Other actions
        self.clear_shortcut = QShortcut(QKeySequence("C"), self)
        self.clear_shortcut.activated.connect(self.clear_points)
        
        self.erase_shortcut = QShortcut(QKeySequence("E"), self)
        self.erase_shortcut.activated.connect(self.toggle_erase_mode)
        
        self.save_obj_shortcut = QShortcut(QKeySequence("Return"), self)
        self.save_obj_shortcut.activated.connect(self.save_current_object)
        
        self.open_shortcut = QShortcut(QKeySequence("O"), self)
        self.open_shortcut.activated.connect(self.open_file)
        
        self.help_shortcut = QShortcut(QKeySequence("H"), self)
        self.help_shortcut.activated.connect(self.show_shortcuts)

    def show_shortcuts(self):
        dialog = ShortcutsDialog(self)
        dialog.exec()

    def toggle_erase_mode(self):
        self.erase_mode = not self.erase_mode
        self.erase_btn.setChecked(self.erase_mode)
        if self.erase_mode:
            self.statusBar().showMessage("Erase mode active")
        else:
            self.statusBar().clearMessage()

    def undo(self):
        if not self.points:
            return
            
        self.redo_stack.append(self.points[-1])
        self.points.pop()
        self.update_mask()

    def redo(self):
        if not self.redo_stack:
            return
            
        self.points.append(self.redo_stack.pop())
        self.update_mask()

    def init_sam_model(self):
        # Download the model checkpoint if not exists
        model_type = "vit_h"
        checkpoint = "sam_vit_h_4b8939.pth"
        
        if not os.path.exists(checkpoint):
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            print("Downloading SAM model checkpoint...")
            urllib.request.urlretrieve(url, checkpoint)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_list = []
            directory = os.path.dirname(file_path)
            for f in os.listdir(directory):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_list.append(os.path.join(directory, f))
            self.image_list.sort()
            self.current_image_index = self.image_list.index(file_path)
            self.load_image(file_path)

    def load_image(self, image_path):
        self.image_path = image_path
        self.current_image = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(self.current_image)
        self.points = []
        self.current_mask = None
        self.finished_masks = []  # Clear finished masks
        
        # Initialize annotations for this image if not exists
        if image_path not in self.annotations:
            self.annotations[image_path] = []
        else:
            # Load existing annotations and recreate masks
            self.load_existing_annotations()
            
        self.display_image()

    def load_existing_annotations(self):
        
        if not self.image_path:
            return
            
        # Load existing annotations from the json file
        labels_dir = os.path.join(os.path.dirname(self.image_path), 'labels')
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        json_path = os.path.join(labels_dir, f"{base_name}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotations = json.load(f)
                
            # Recreate masks from polygons
            for annotation in annotations:
                mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                polygon_points = np.array(annotation['polygon'], dtype=np.int32)
                cv2.fillPoly(mask, [polygon_points], 1)
                # Generate a random color for this mask
                color = np.random.randint(50, 200, size=3).tolist()
                self.finished_masks.append((mask, color))

    def load_existing_annotations2(self, points):
        if not self.image_path:
            return
            
        # Load existing annotations from the json file
        labels_dir = os.path.join(os.path.dirname(self.image_path), 'labels')
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        json_path = os.path.join(labels_dir, f"{base_name}.json")
        
        self.finished_masks = []
        

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            print("loaded annotastion", len(annotations))
            # Recreate masks from polygons
            for annotation in annotations:
                mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                polygon_points = np.array(annotation['polygon'], dtype=np.int32)
                print(points)

                point_in_polygon = cv2.pointPolygonTest(polygon_points, points[0], False)
                if point_in_polygon >= 0:
                    print("YES: Point lies inside or on the boundary of a polygon.")
                else:
                    cv2.fillPoly(mask, [polygon_points], 1)
                    # Generate a random color for this mask
                    color = np.random.randint(50, 200, size=3).tolist()
                    self.finished_masks.append((mask, color))

    def display_image(self):
        if self.current_image is None:
            return

        display_image = self.current_image.copy()

        # Draw all finished masks with their colors
         
        for mask, color in self.finished_masks:
            mask_overlay = np.zeros_like(display_image)
            mask_overlay[mask == 1] = color
            display_image = cv2.addWeighted(display_image, 1, mask_overlay, 0.5, 0)

        # Draw current mask if exists
        if self.current_mask is not None:
            mask_overlay = np.zeros_like(display_image)
            mask_overlay[self.current_mask] = [0, 255, 0]  # Green overlay for current mask
            display_image = cv2.addWeighted(display_image, 1, mask_overlay, 0.5, 0)

        # Draw points
        for point in self.points:
            color = (255, 0, 0) if point[2] == 1 else (0, 0, 255)  # Blue for erase mode
            cv2.circle(display_image, (point[0], point[1]), 5, color, -1)

        # Convert to QPixmap and display
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.set_base_pixmap(pixmap)

    def on_image_click(self, event):
        if self.current_image is None or event.button() == Qt.MouseButton.MiddleButton:
            return

        # Get click coordinates with proper translation
        image_pos = self.image_label.get_image_coordinates(event.pos())
        if image_pos is None:
            return

        # Clear redo stack when new action is performed
        self.redo_stack.clear()
        
        # Add point with appropriate label
        self.points.append([image_pos.x(), image_pos.y()])

        if self.erase_mode:
            self.points[-1].append(0)  # 0 for background

        elif (event.button() == Qt.MouseButton.RightButton):
            self.current_mask = None
            self.finished_masks = [] 
            self.load_existing_annotations2(self.points)
            self.points = []
        else:
            self.points[-1].append(1)  # 1 for foreground
            
        self.update_mask()

    def update_mask(self):
        if not self.points:
            self.current_mask = None
            self.display_image()
            return

        input_points = np.array([p[:2] for p in self.points])
        input_labels = np.array([p[2] for p in self.points])
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Select the mask with highest score
        self.current_mask = masks[np.argmax(scores)]
        self.display_image()

    def mask_to_polygon(self, mask):
        """Convert binary mask to polygon points"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the contour with maximum area
        if not contours:
            return None
            
        max_contour = max(contours, key=cv2.contourArea)
        
        # Simplify the contour to reduce number of points
        epsilon = 0.005 * cv2.arcLength(max_contour, True)
        approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # Convert to list of points
        polygon = approx_contour.reshape(-1, 2).tolist()
        return polygon

    def save_current_object(self):
        if self.current_mask is None or self.image_path is None:
            return

        # Convert mask to polygon
        polygon = self.mask_to_polygon(self.current_mask)
        if polygon is None:
            QMessageBox.warning(self, "Warning", "Failed to create polygon from mask")
            return

        # Create annotation
        annotation = {
            'label': self.label_combo.currentText(),
            'polygon': polygon  # Save polygon points instead of mask
        }

        # Add to annotations
        if self.image_path not in self.annotations:
            self.annotations[self.image_path] = []
        self.annotations[self.image_path].append(annotation)
        
        # Generate random color for this mask
        color = np.random.randint(50, 200, size=3).tolist()
        
        # Add to finished masks
        self.finished_masks.append((self.current_mask.copy(), color))
        
        # Save the annotation immediately
        self.save_current_image_annotation()
        
        # Clear current selection but keep finished masks visible
        self.clear_points()
        QMessageBox.information(self, "Success", "Object saved successfully!")

    def save_current_image_annotation(self):
        if not self.image_path or self.image_path not in self.annotations:
            return

        # Create labels directory if it doesn't exist
        labels_dir = os.path.join(os.path.dirname(self.image_path), 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        # Get base filename without extension and create json path
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        json_path = os.path.join(labels_dir, f"{base_name}.json")

        # Save annotations for current image
        with open(json_path, 'w') as f:
            json.dump(self.annotations[self.image_path], f, indent=2)

    def save_annotations(self):
        
        # Save all annotations
        for image_path in self.annotations:
            # print(image_path)
            if self.annotations[image_path]:  # Only save if there are annotations
                labels_dir = os.path.join(os.path.dirname(image_path), 'labels')
                os.makedirs(labels_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                json_path = os.path.join(labels_dir, f"{base_name}.json")
                
                with open(json_path, 'w') as f:
                    json.dump(self.annotations[image_path], f, indent=2)
        
        QMessageBox.information(self, "Success", "All annotations saved successfully!")

    def clear_points(self):
        if self.points:
            self.undo_stack.append(self.points.copy())
        self.points = []
        self.redo_stack.clear()
        self.current_mask = None
        self.display_image()

    def next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image(self.image_list[self.current_image_index])

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.image_list[self.current_image_index])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec())