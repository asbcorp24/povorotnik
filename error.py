import cv2
import numpy as np
import time
from tkinter import Tk, Label, Button, filedialog, Toplevel, Frame, Canvas, Scale, HORIZONTAL, StringVar, OptionMenu
from PIL import Image, ImageTk
from skimage import segmentation, measure
from skimage import img_as_float
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.transform import resize
from skimage.measure import label, regionprops  # Добавили импорт функции label+
from skimage.metrics import structural_similarity as ssim
class ImageAlignerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Image Aligner")

        # Main frame setup
        self.main_frame = Frame(master)
        self.main_frame.pack(fill="both", expand=True)

        self.label = Label(self.main_frame, text="Choose images to align", font=("Arial", 16))
        self.label.pack(pady=10)

        # Buttons for loading images
        self.choose_original_button = Button(self.main_frame, text="Choose Original Image", command=self.load_original_image, bg="lightblue", font=("Arial", 12))
        self.choose_original_button.pack(pady=5)

        self.choose_target_button = Button(self.main_frame, text="Choose Target Image", command=self.load_target_image, bg="lightgreen", font=("Arial", 12))
        self.choose_target_button.pack(pady=5)

        self.align_button = Button(self.main_frame, text="Align Images", command=self.align_images, state="disabled", bg="orange", font=("Arial", 12))
        self.align_button.pack(pady=10)

        self.compare_button = Button(self.main_frame, text="Compare Images", command=self.compare_images, bg="yellow", font=("Arial", 12), state="disabled")
        self.compare_button.pack(pady=5)
        # New button for improved comparison
        self.improved_compare_button = Button(self.main_frame, text="Improved Compare",
                                              command=self.improved_compare_images, bg="purple", font=("Arial", 12),
                                             )
        # New button for improved comparison
        self.roi_compare_button = Button(self.main_frame, text="Compare Within ROI", command=self.compare_within_roi,
                                         bg="purple", font=("Arial", 12))

        self.roi_compare_button = Button(self.main_frame, text="Compare Within ROI2", command=self.compare_within_roi2,
                                         bg="purple", font=("Arial", 12) )
        self.ssim_button = Button(self.main_frame, text="Compare with SSIM", command=self.compare_with_ssim,
                                  bg="purple", font=("Arial", 12) )
        self.ssim_button.pack(pady=5)

        self.active_contour_button = Button(self.main_frame, text="Compare with Active Contour",
                                            command=self.compare_with_active_contour, bg="purple", font=("Arial", 12),
                                             )
        self.active_contour_button.pack(pady=5)

        self.roi_compare_button.pack(pady=5)

        self.roi_compare_button.pack(pady=5)

        self.improved_compare_button.pack(pady=5)
        self.save_button = Button(self.main_frame, text="Save Result", command=self.save_result, bg="lightgreen", font=("Arial", 12))
        self.save_button.pack(pady=5)

        # Rotation slider
        self.rotation_label = Label(self.main_frame, text="Rotate Original Image:", font=("Arial", 12))
        self.rotation_label.pack(pady=5)

        self.rotation_slider = Scale(self.main_frame, from_=0, to=360, orient=HORIZONTAL, command=self.rotate_original_image, length=400)
        self.rotation_slider.pack(pady=5)

        # Ignore color selection
        self.ignore_color_label = Label(self.main_frame, text="Ignore Color for Comparison:", font=("Arial", 12))
        self.ignore_color_label.pack(pady=5)

        self.ignore_color_var = StringVar(self.main_frame)
        self.ignore_color_var.set("None")
        self.ignore_color_menu = OptionMenu(self.main_frame, self.ignore_color_var, "None", "Red", "Green", "Blue")
        self.ignore_color_menu.pack(pady=5)

        self.original_image = None
        self.original_image_rotated = None
        self.target_image = None
        self.result_image = None
        self.original_window = None
        self.target_window = None
        self.result_window = None
        self.matches_window = None
        self.comparison_window = None
    def compare_with_ssim(self):
        if self.original_image_rotated is None or self.result_image is None:
            print("Images are not ready for comparison.")
            return

        # Преобразуем изображения в серое
        original_gray = cv2.cvtColor(self.original_image_rotated, cv2.COLOR_BGR2GRAY)
        modified_gray = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2GRAY)

        # Вычисляем SSIM между изображениями
        ssim_value, ssim_map = ssim(original_gray, modified_gray, full=True)

        # Отображаем карту различий
        ssim_map = (ssim_map * 255).astype(np.uint8)
        highlighted_image = cv2.applyColorMap(ssim_map, cv2.COLORMAP_JET)

        self.display_image(highlighted_image, "SSIM Difference Map", "ssim_comparison_window")


    def preprocess_image(self, image):
        # Преобразуем в серый и применяем сглаживание
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def active_contour(self, image):
        # Преобразуем изображение в плавающие значения
        image_float = img_as_float(image)

        # Применение Canny edge detector для выделения краев
        edges = canny(image_float)

        # Автоматическое нахождение области интереса через контуры
        labeled_image = label(edges)  # Обнаруживаем контуры
        regions = regionprops(labeled_image)  # Получаем регионы объектов

        # Используем первый регион для инициализации активного контура
        if regions:
            minr, minc, maxr, maxc = regions[0].bbox
            init = np.array([[minc, minr], [maxc, minr], [maxc, maxr], [minc, maxr]])
            init = np.reshape(init, (-1, 2))  # Преобразуем в нужный формат для активного контура
        else:
            # Если нет обнаруженных объектов, инициализируем контур вручную
            init = np.array([[[50, 50], [250, 50], [250, 250], [50, 250]]])  # Прямоугольник
            init = np.reshape(init, (-1, 2))  # Преобразуем в нужный формат

        # Применение метода активных контуров
        active_contour_result = segmentation.active_contour(edges, init, alpha=0.015, beta=10, gamma=0.001,
                                                            max_num_iter=250)

        return active_contour_result

    def compare_with_active_contour(self):
        if self.original_image_rotated is None or self.result_image is None:
            print("Images are not ready for comparison.")
            return

        # Преобразуем изображения в серое
        original_processed = self.preprocess_image(self.original_image_rotated)
        modified_processed = self.preprocess_image(self.result_image)

        # Применяем активные контуры
        contour_original = self.active_contour(original_processed)
        contour_modified = self.active_contour(modified_processed)

        # Отображаем результат
        cv2.polylines(self.original_image_rotated, [contour_original.astype(np.int32)], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        cv2.polylines(self.result_image, [contour_modified.astype(np.int32)], isClosed=True, color=(0, 255, 0),
                      thickness=2)

        # Показываем изображения с выделенными контурами
        self.display_image(self.original_image_rotated, "Original with Active Contour", "original_with_contour_window")
        self.display_image(self.result_image, "Modified with Active Contour", "modified_with_contour_window")

    def compare_within_roi2(self):
        if self.original_image_rotated is None or self.result_image is None:
            print("Images are not ready for comparison.")
            return

        # Преобразуем изображения в серый формат
        original_image_gray = cv2.cvtColor(self.original_image_rotated, cv2.COLOR_BGR2GRAY)
        result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2GRAY)

        # Применяем SIFT для выделения ключевых точек
        sift = cv2.SIFT_create()
        keypoints1, _ = sift.detectAndCompute(original_image_gray, None)
        keypoints2, _ = sift.detectAndCompute(result_image_gray, None)

        # Создаем маски для выделения ROI (регионов интереса) вокруг ключевых точек
        mask1 = np.zeros_like(original_image_gray)
        mask2 = np.zeros_like(result_image_gray)

        for kp in keypoints1:
            x, y = kp.pt
            cv2.circle(mask1, (int(x), int(y)), 50, 255, -1)  # Радиус вокруг ключевых точек
        for kp in keypoints2:
            x, y = kp.pt
            cv2.circle(mask2, (int(x), int(y)), 50, 255, -1)

        # Применяем маски для выделения только интересующих областей
        roi_original = cv2.bitwise_and(original_image_gray, original_image_gray, mask=mask1)
        roi_result = cv2.bitwise_and(result_image_gray, result_image_gray, mask=mask2)

        # Вычисление абсолютной разницы и выделение различий
        diff = cv2.absdiff(roi_original, roi_result)
        _, threshold_diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # Морфологическое открытие для удаления мелких шумов
        kernel = np.ones((5, 5), np.uint8)
        threshold_diff = cv2.morphologyEx(threshold_diff, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        highlighted_result = self.result_image.copy()
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(highlighted_result, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.display_image(highlighted_result, "Differences Inside ROI", "roi_comparison_window")

    def compare_within_roi(self):
        if self.original_image_rotated is None or self.result_image is None:
            print("Images are not ready for comparison.")
            return

        # Преобразуем изображения в серый формат
        original_image_gray = cv2.cvtColor(self.original_image_rotated, cv2.COLOR_BGR2GRAY)
        result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2GRAY)

        # Применяем SIFT для выделения ключевых точек
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(original_image_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(result_image_gray, None)

        # Создаем маски для выделения ROI (регионов интереса) вокруг ключевых точек
        mask1 = np.zeros_like(original_image_gray)
        mask2 = np.zeros_like(result_image_gray)
        for kp in keypoints1:
            x, y = kp.pt
            cv2.circle(mask1, (int(x), int(y)), 50, 255, -1)  # Радиус вокруг ключевых точек
        for kp in keypoints2:
            x, y = kp.pt
            cv2.circle(mask2, (int(x), int(y)), 50, 255, -1)

        # Применяем маски для выделения только интересующих областей
        roi_original = cv2.bitwise_and(original_image_gray, original_image_gray, mask=mask1)
        roi_result = cv2.bitwise_and(result_image_gray, result_image_gray, mask=mask2)

        # Вычисление абсолютной разницы внутри ROI
        diff = cv2.absdiff(roi_original, roi_result)
        _, threshold_diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Выделяем различия внутри ROI
        highlighted_result = self.result_image.copy()
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(highlighted_result, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.display_image(highlighted_result, "Differences Inside ROI", "roi_comparison_window")

    def improved_compare_images(self):
        if self.original_image_rotated is None or self.result_image is None:
            print("Images are not ready for comparison.")
            return

        # Приведение изображений к одинаковому размеру
        height, width = self.result_image.shape[:2]
        resized_original_image = cv2.resize(self.original_image_rotated, (width, height))

        # Преобразование в оттенки серого
        original_image = cv2.cvtColor(resized_original_image, cv2.COLOR_BGR2GRAY)
        result_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2GRAY)

        # SIFT for feature matching
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
        keypoints2, descriptors2 = sift.detectAndCompute(result_image, None)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=50))
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Draw matches
        matched_image = cv2.drawMatches(self.original_image_rotated, keypoints1, self.result_image, keypoints2,
                                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        self.display_image(matched_image, "Improved Keypoint Matches", "improved_matches_window")

        # Compute absolute difference and highlight it
        diff = cv2.absdiff(resized_original_image, self.result_image)

        # Ensure images have the same number of channels
        if len(diff.shape) == 3 and diff.shape[2] == 3:  # If 3 channels (BGR)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff

        _, threshold_diff = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        highlighted_result = self.result_image.copy()
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(highlighted_result, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.display_image(highlighted_result, "Improved Differences Highlighted", "improved_comparison_window")

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image, _ = self.resize_for_processing(cv2.imread(file_path))
            self.original_image_rotated = self.original_image.copy()
            self.display_image(self.original_image_rotated, "Original Image", "original_window")
            self.check_ready()

    def load_target_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.target_image, _ = self.resize_for_processing(cv2.imread(file_path))
            self.display_image(self.target_image, "Target Image", "target_window")
            self.check_ready()

    def check_ready(self):
        if self.original_image is not None and self.target_image is not None:
            self.align_button["state"] = "normal"
            self.compare_button["state"] = "normal"

    def rotate_original_image(self, angle):
        if self.original_image is not None:
            angle = float(angle)
            height, width = self.original_image.shape[:2]
            center = (width // 2, height // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_matrix[0, 2] += (bound_w / 2) - center[0]
            rotation_matrix[1, 2] += (bound_h / 2) - center[1]

            self.original_image_rotated = cv2.warpAffine(self.original_image, rotation_matrix, (bound_w, bound_h))
            self.display_image(self.original_image_rotated, "Rotated Original Image", "original_window")

    def display_image(self, img, title, window_attr):
        max_width = 600
        height, width = img.shape[:2]
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img, (new_width, new_height))

        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        window = getattr(self, window_attr, None)
        if window and window.winfo_exists():
            canvas = getattr(self, f"{window_attr}_canvas")
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk
        else:
            window = Toplevel(self.master)
            window.title(title)
            setattr(self, window_attr, window)

            canvas = Canvas(window, width=new_width, height=new_height, bg="white")
            canvas.pack()
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            setattr(self, f"{window_attr}_canvas", canvas)
            canvas.image = img_tk

    def align_images(self):
        start_time = time.time()
        gray_original = cv2.cvtColor(self.original_image_rotated, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray_original, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_target, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        matched_image = cv2.drawMatches(self.original_image_rotated, keypoints1, self.target_image, keypoints2, good_matches, None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))
        self.display_image(matched_image, "Keypoint Matches", "matches_window")

        height, width = self.original_image_rotated.shape[:2]
        self.result_image = cv2.warpPerspective(self.target_image, matrix, (width, height))
        self.display_image(self.result_image, "Aligned Result", "result_window")

        end_time = time.time()
        print(f"Alignment completed in {end_time - start_time:.2f} seconds.")

    def compare_images(self):
        if self.original_image_rotated is None or self.result_image is None:
            print("Images are not ready for comparison.")
            return

        ignore_color = self.ignore_color_var.get().lower()

        # Convert images to HSV for color filtering
        hsv_original = cv2.cvtColor(self.original_image_rotated, cv2.COLOR_BGR2HSV)
        hsv_result = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2HSV)

        if ignore_color == "red":
            lower_bound = np.array([0, 50, 50])
            upper_bound = np.array([10, 255, 255])
            mask_original = cv2.inRange(hsv_original, lower_bound, upper_bound)
            mask_result = cv2.inRange(hsv_result, lower_bound, upper_bound)
        elif ignore_color == "green":
            lower_bound = np.array([40, 50, 50])
            upper_bound = np.array([80, 255, 255])
            mask_original = cv2.inRange(hsv_original, lower_bound, upper_bound)
            mask_result = cv2.inRange(hsv_result, lower_bound, upper_bound)
        elif ignore_color == "blue":
            lower_bound = np.array([100, 50, 50])
            upper_bound = np.array([140, 255, 255])
            mask_original = cv2.inRange(hsv_original, lower_bound, upper_bound)
            mask_result = cv2.inRange(hsv_result, lower_bound, upper_bound)
        else:
            mask_original = mask_result = np.zeros_like(hsv_original[:, :, 0])

        # Exclude ignored color
        filtered_original = cv2.bitwise_and(self.original_image_rotated, self.original_image_rotated,
                                            mask=cv2.bitwise_not(mask_original))
        filtered_result = cv2.bitwise_and(self.result_image, self.result_image, mask=cv2.bitwise_not(mask_result))

        # Compute the absolute difference
        difference = cv2.absdiff(filtered_original, filtered_result)
        gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, threshold_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

        # Highlight differences
        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        highlighted_result = self.result_image.copy()

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(highlighted_result, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Create heatmap of differences
        diff = cv2.absdiff(self.original_image_rotated, self.result_image)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        # Display the highlighted differences and heatmap
        self.display_image(highlighted_result, "Differences Highlighted", "comparison_window")
        self.display_image(heatmap, "Heatmap of Differences", "heatmap_window")

    def save_result(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.result_image)

    def resize_for_processing(self, image, max_size=1000):
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            return resized, scale
        return image, 1.0

if __name__ == "__main__":
    root = Tk()
    app = ImageAlignerApp(root)
    root.mainloop()
