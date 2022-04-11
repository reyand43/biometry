from re import S
from tkinter import (LEFT, TOP, Button, Canvas, E, Frame, Label, Tk, W,
                     filedialog)

import cv2
from domain.recognition import recognition
from PIL import Image, ImageTk


class App(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Picture Artist Classification")
        self.geometry("1500x1000")

        self.interface_frame = Frame(self)
        self.canvas_frame = Frame(self)
        self.results_frame = Frame(self)

        
        self.interface_frame.pack(side=TOP, anchor=W)
        self.results_frame.pack(side=TOP, anchor=W)
        self.canvas_frame.pack(side=TOP, anchor=W)

    # ========= UX =========
        self.label = Label(
            self.interface_frame,
            text="Art classificator",
            font="Times 20"
        )
        self.label.pack(side=TOP, anchor=W, padx=15, pady=7)

        self.photo = Button(
            self.interface_frame,
            text="Upload",
            width=20,
            command=lambda: self.upload_image()
        )
        self.photo.pack(side=LEFT, anchor=W, padx=15, pady=7)

        self.recognition = Button(
            self.interface_frame,
            text="Classify",
            width=20,
            command=lambda: self.classifier()
            )
        self.recognition.pack(side=LEFT, anchor=W, padx=15, pady=7)


        self.Harris_label = Label(self.results_frame, text="Harris: ", font="20")
        self.Harris_label.pack(side=LEFT, anchor=W)
        self.Harris_value = Label(self.results_frame, text="", font="20")
        self.Harris_value.pack(side=LEFT, anchor=W)

        self.BRISK_label = Label(self.results_frame, text="BRISK: ", font="20")
        self.BRISK_label.pack(side=LEFT, anchor=W)
        self.BRISK_value = Label(self.results_frame, text="", font="20")
        self.BRISK_value.pack(side=LEFT, anchor=W)

        self.FAST_label = Label(self.results_frame, text="FAST: ", font="20")
        self.FAST_label.pack(side=LEFT, anchor=W)
        self.FAST_value = Label(self.results_frame, text="", font="20")
        self.FAST_value.pack(side=LEFT, anchor=W)

    # ========= Canvas =========
        self.canvas = Canvas(self.canvas_frame, width=500, height=500)
        self.canvas.pack(side=TOP)

        self.image = None
        self.result = []


    def upload_image(self) -> None:
        global image
        filename = filedialog.askopenfilename(title="upload")
        self.image = cv2.imread(filename)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(self.image)
        image.save("./data/uploaded/photo.png")
        img=image.resize((450, 350), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(img)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.canvas.create_image(0, 0, anchor='nw', image=image)

    def classifier(self) -> None:
        if self.image is None:
            raise Exception("Image not uploaded")
        mark, marks = recognition([self.image])
        self.Harris_value.config(text=marks[0])
        self.BRISK_value.config(text=marks[1])
        self.FAST_value.config(text=marks[2])

