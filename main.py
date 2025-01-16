import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk
from tkinter import Toplevel, Label
class ImageAlignerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Aligner")

        self.label = Label(master, text="Choose images to align")
        self.label.pack()

        self.choose_original_button = Button(master, text="Choose Original Image", command=self.load_original_image)
        self.choose_original_button.pack()

        self.choose_target_button = Button(master, text="Choose Target Image", command=self.load_target_image)
        self.choose_target_button.pack()

        self.align_button = Button(master, text="Align Images", command=self.align_images, state="disabled")
        self.align_button.pack()

        self.original_image = None
        self.target_image = None
        self.result_image = None

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, "Original Image")
            self.check_ready()

    def load_target_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.target_image = cv2.imread(file_path)
            self.display_image(self.target_image, "Target Image")
            self.check_ready()

    def check_ready(self):
        if self.original_image is not None and self.target_image is not None:
            self.align_button["state"] = "normal"

    def display_image(self, img, title):
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

        # Создаем новое окно для отображения изображения
        window = Toplevel(self.master)
        window.title(title)

        label = Label(window, image=img_tk)
        label.image = img_tk  # Сохраняем ссылку на изображение, чтобы оно не удалилось
        label.pack()


    def align_images(self):
        gray_original = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
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

        # Нарисуем ключевые точки на изображении
        matched_image = cv2.drawMatches(
            self.original_image, keypoints1,
            self.target_image, keypoints2,
            best_matches, None,
            matchColor=(0, 255, 0),  # Зеленый цвет для линий
            singlePointColor=(0, 255, 0)  # Зеленый цвет для точек
        )

        # Показать изображение с совпадениями
        self.display_image(matched_image, "Keypoint Matches")

        # Вычисление матрицы гомографии
        matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

        height, width = self.original_image.shape[:2]
        self.result_image = cv2.warpPerspective(self.target_image, matrix, (width, height))

        # Показать выровненное изображение
        self.display_image(self.result_image, "Aligned Result")


if __name__ == "__main__":
    root = Tk()
    app = ImageAlignerApp(root)
    root.mainloop()
