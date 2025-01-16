import cv2
import numpy as np
import time
from tkinter import Tk, Label, Button, filedialog, Toplevel, Frame, Canvas, Scale, HORIZONTAL
from PIL import Image, ImageTk

class ImageAlignerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Вращалка")

        # Main frame setup
        self.main_frame = Frame(master)
        self.main_frame.pack(fill="both", expand=True)

        self.label = Label(self.main_frame, text="Выберите что вращать", font=("Arial", 16))
        self.label.pack(pady=10)

        # Buttons for loading images
        self.choose_original_button = Button(self.main_frame, text="Оргинал", command=self.load_original_image, bg="lightblue", font=("Arial", 12))
        self.choose_original_button.pack(pady=5)

        self.choose_target_button = Button(self.main_frame, text="Че вращаем", command=self.load_target_image, bg="lightgreen", font=("Arial", 12))
        self.choose_target_button.pack(pady=5)

        self.align_button = Button(self.main_frame, text="повернуть ", command=self.align_images, state="disabled", bg="orange", font=("Arial", 12))
        self.align_button.pack(pady=10)

        self.save_button = Button(self.main_frame, text="Сохранить результ", command=self.save_result, bg="lightgreen", font=("Arial", 12))
        self.save_button.pack(pady=5)

        # Rotation slider
        self.rotation_label = Label(self.main_frame, text="Повернутое изображение:", font=("Arial", 12))
        self.rotation_label.pack(pady=5)

        self.rotation_slider = Scale(self.main_frame, from_=0, to=360, orient=HORIZONTAL, command=self.rotate_original_image, length=400)
        self.rotation_slider.pack(pady=5)

        self.original_image = None
        self.original_image_rotated = None
        self.target_image = None
        self.result_image = None
        self.original_window = None
        self.target_window = None
        self.result_window = None
        self.matches_window = None
        self.comparison_window = None

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image, _ = self.resize_for_processing(cv2.imread(file_path))
            self.original_image_rotated = self.original_image.copy()
            self.display_image(self.original_image_rotated, "Оригинал", "original_window")
            self.check_ready()

    def load_target_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.target_image, _ = self.resize_for_processing(cv2.imread(file_path))
            self.display_image(self.target_image, "че вращаем", "target_window")
            self.check_ready()

    def check_ready(self):
        if self.original_image is not None and self.target_image is not None:
            self.align_button["state"] = "normal"

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
            self.display_image(self.original_image_rotated, "Повернутое", "original_window")

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
        self.display_image(matched_image, "Токчки", "matches_window")

        height, width = self.original_image_rotated.shape[:2]
        self.result_image = cv2.warpPerspective(self.target_image, matrix, (width, height))
        self.display_image(self.result_image, "поревнутый резулсь", "result_window")

        comparison_image = self.side_by_side(self.original_image_rotated, self.result_image)
        self.display_image(comparison_image, "Сравнивание", "comparison_window")

        end_time = time.time()
        print(f"Все сделал за {end_time - start_time:.2f} секунды.")

    def save_result(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG ", "*.jpg"), ("PNG ", "*.png")])
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

    def side_by_side(self, image1, image2):
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]
        result = np.zeros((height, width, 3), dtype=np.uint8)
        result[:image1.shape[0], :image1.shape[1]] = image1
        result[:image2.shape[0], image1.shape[1]:] = image2
        return result

if __name__ == "__main__":
    root = Tk()
    app = ImageAlignerApp(root)
    root.mainloop()
