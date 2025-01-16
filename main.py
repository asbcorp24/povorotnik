import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Toplevel, Frame, Canvas, Scrollbar, Scale, HORIZONTAL
from PIL import Image, ImageTk

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

        # Rotation slider
        self.rotation_label = Label(self.main_frame, text="Rotate Original Image:", font=("Arial", 12))
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

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image_rotated = self.original_image.copy()
            self.display_image(self.original_image_rotated, "Original Image", "original_window")
            self.check_ready()

    def load_target_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.target_image = cv2.imread(file_path)
            self.display_image(self.target_image, "Target Image", "target_window")
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
            self.original_image_rotated = cv2.warpAffine(self.original_image, rotation_matrix, (width, height))
            self.display_image(self.original_image_rotated, "Rotated Original Image", "original_window")

    def display_image(self, img, title, window_attr):
        # Масштабируем изображение до ширины 600 пикселей с сохранением пропорций
        max_width = 600
        height, width = img.shape[:2]
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img, (new_width, new_height))

        # Преобразуем изображение для отображения в Tkinter
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Проверяем существование окна
        window = getattr(self, window_attr, None)
        if window and window.winfo_exists():
            # Если окно уже существует, обновляем содержимое
            canvas = getattr(self, f"{window_attr}_canvas")
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk
        else:
            # Если окна нет, создаем его
            window = Toplevel(self.master)
            window.title(title)
            setattr(self, window_attr, window)

            canvas = Canvas(window, width=new_width, height=new_height, bg="white")
            canvas.pack()
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            setattr(self, f"{window_attr}_canvas", canvas)

            # Сохраняем изображение, чтобы предотвратить его удаление
            canvas.image = img_tk

    def align_images(self):
        gray_original = cv2.cvtColor(self.original_image_rotated, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray_original, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray_target, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        best_matches = matches[:50]

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        matched_image = cv2.drawMatches(
            self.original_image_rotated, keypoints1,
            self.target_image, keypoints2,
            best_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0)
        )

        self.display_image(matched_image, "Keypoint Matches", "matches_window")

        matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        height, width = self.original_image_rotated.shape[:2]
        self.result_image = cv2.warpPerspective(self.target_image, matrix, (width, height))

        self.display_image(self.result_image, "Aligned Result", "result_window")

if __name__ == "__main__":
    root = Tk()
    app = ImageAlignerApp(root)
    root.mainloop()
