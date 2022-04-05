from pickle import FRAME
import random
from tkinter import (LEFT, RIGHT, TOP, Button, Canvas, Entry, Frame, Label,
                     OptionMenu, StringVar, W, ttk)

from core.config.config import ALL_METHODS, DATA_PATH
from core.recognition import parallel_recognition, recognition
from PIL import Image, ImageTk
from core.research import parallel_system_research, research

DB_VALUES = ["ORL", "ORL_mask", "ORL_fawkes"]

class ExperimentFrame(Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        


        self.settings_frame = Frame(self)
        self.result_frame = Frame(self)

        self.db_label = Label(self.settings_frame, text="DB name: ")
        self.db_value = StringVar()
        self.db_value.set(DB_VALUES[0])
        self.db_value_drop = OptionMenu(
            self.settings_frame,
            self.db_value,
            *DB_VALUES,
        )
        self.db_value_drop.configure(width=10)

        self.consistent_frame = Frame(self.settings_frame, highlightbackground="blue", highlightthickness=2)
        self.parallel_frame = Frame(self.settings_frame, highlightbackground="blue", highlightthickness=2)
        self.consistent_frame2 = Frame(self.settings_frame, highlightbackground="blue", highlightthickness=2)
        self.parallel_frame2 = Frame(self.settings_frame, highlightbackground="blue", highlightthickness=2)
        self.canvas_frame = Frame(self.result_frame)

    # =============== ПОСЛЕДОВАТЕЛЬНАЯ СИСТЕМА ===============

        self.cons_label = Label(
            self.consistent_frame,
            text="Classifier work demo",
            font='Arial 20 bold',
        )

        self.frame1 = Frame(self.consistent_frame)
        self.frame2 = Frame(self.consistent_frame)
        self.frame3 = Frame(self.consistent_frame)
        self.method_label = Label(self.frame1, text="Method: ")

        self.method = StringVar()
        self.method.set(ALL_METHODS[1])
        self.method_drop = OptionMenu(
            self.frame1,
            self.method,
            *ALL_METHODS,
        )
        self.method_drop.configure(width=10)

        # Параметр метода
        self.param_label = Label(self.frame1, text="Parameter value: ")
        self.p_entry = Entry(self.frame1, width=7)

        # Номер, с которого будут браться шаблоны для каждого человека
        self.from_templ_label = Label(self.frame2, text="Train start: ")
        self.from_template_entry = Entry(self.frame2, width=7)

        # Номер, по который будут браться шаблоны для каждого человека
        self.to_templ_label = Label(self.frame2, text="Train end: ")
        self.to_template_entry = Entry(self.frame2, width=7)
        self.score_label = Label(self.frame3, text="Accuracy:")
        self.score_result = Label(self.frame3, text="", font='Arial 15 bold')

        # Список классифицируемых изображений
        self.result_images = []

        # Список содержащий по шаблону для
        # каждого из классифицируемых изображений
        self.templates = []

        # Запуск исследования
        self.run_but = Button(
            self.frame3,
            text="Run",
            command=lambda: self.consistent_experiment(),
        )


    # =============== ПАРАЛЛЕЛЬНАЯ СИСТЕМА ===============
        self.parallel_label = Label(
            self.parallel_frame,
            text="Parallel system",
            font='Arial 20 bold',
        )

        # ПАРАМЕТРЫ
        self.frame4 = Frame(self.parallel_frame)
        self.frame5 = Frame(self.parallel_frame)
        self.frame6 = Frame(self.parallel_frame)
        self.frame7 = Frame(self.parallel_frame)

        # Histogram
        self.hist_label = Label(self.frame4, text="Histogram: ")
        self.hist_entry = Entry(self.frame4, width=7)

        # Scale
        self.scale_label = Label(self.frame4, text="Scale: ")
        self.scale_entry = Entry(self.frame4, width=7)

        # Gradient
        self.gradient_label = Label(self.frame5, text="Gradient: ")
        self.gradient_entry = Entry(self.frame5, width=7)

        # DFT
        self.dft_label = Label(self.frame5, text="DFT: ")
        self.dft_entry = Entry(self.frame5, width=7)

        # DCT
        self.dct_label = Label(self.frame6, text="DCT: ")
        self.dct_entry = Entry(self.frame6, width=7)

        # Число шаблонов
        self.templ_num_label = Label(self.frame6, text="Train templates number: ")
        self.templ_num_entry = Entry(self.frame6, width=7)

        # Запуск параллельного исследования
        self.parallel_button = Button(
            self.frame7,
            text="Run",
            command=lambda: self.parallel_experiment(),
        )

        self.canvas = Canvas(self.canvas_frame, width=1000, height=900)

    # ======================= ПОСЛЕДОВАТЕЛЬНАЯ СИСТЕМА 2 =======================

        self.cons_label2 = Label(
            self.consistent_frame2,
            text="Dependence of accuracy on parameter values",
            font='Arial 20 bold'
        )

        # Метод извлечения признаков


        self.method_label2 = Label(self.consistent_frame2, text="Method")

        self.method2 = StringVar()
        self.method2.set(ALL_METHODS[1])

        self.method_drop2 = OptionMenu(
            self.consistent_frame2,
            self.method2,
            *ALL_METHODS
        )
        self.method_drop2.configure(width=10)

        # Список классифицируемых изображений
        self.result_images2 = []

        self.temp_to_label2 = Label(self.consistent_frame2, text="Number of photos: ")
        self.temp_to_entry2 = Entry(self.consistent_frame2, width=7)

        # Запуск исследования
        self.run_but2 = Button(
            self.consistent_frame2, text="Run", command=lambda: self.research()
        )
    
    # ======================= ПАРЛЛЕЛЬНАЯ СИСТЕМА 2 =======================
        self.parallel_label2 = Label(
            self.parallel_frame2,
            text="Parallel system, sample size dependence (CV)",
            font='Arial 20 bold'
        )

        # ПАРАМЕТРЫ

        self.frame8 = Frame(self.parallel_frame2)
        self.frame9 = Frame(self.parallel_frame2)
        self.frame10 = Frame(self.parallel_frame2)

        # Histogram
        self.hist_label2 = Label(self.frame8, text="Histogram: ")
        self.hist_entry2 = Entry(self.frame8, width=7)

        # Scale
        self.scale_label2 = Label(self.frame8, text="Scale: ")
        self.scale_entry2 = Entry(self.frame8, width=7)

        # Gradient
        self.gradient_label2 = Label(self.frame9, text="Gradient: ")
        self.gradient_entry2 = Entry(self.frame9, width=7)

        # DFT
        self.dft_label2 = Label(self.frame9, text="DFT: ")
        self.dft_entry2 = Entry(self.frame9, width=7)

        # DCT
        self.dct_label2 = Label(self.frame10, text="DCT: ")
        self.dct_entry2 = Entry(self.frame10, width=7)

        # Запуск параллельного исследования
        self.parallel_button2 = Button(
            self.frame10,
            text="Run",
            command=lambda: self.parallel_research()
        )


    # =============== МЕСТОПОЛОЖЕНИЕ ВИДЖЕТОВ ===============
        self.settings_frame.pack(side=LEFT, anchor=W)
        self.result_frame.pack(side=LEFT, anchor=W)

        self.consistent_frame.pack(side=TOP, anchor=W)
        self.parallel_frame.pack(side=TOP, anchor=W, pady=20)
        self.consistent_frame2.pack(side=TOP, anchor=W)
        self.parallel_frame2.pack(side=TOP, anchor=W, pady=20)
        self.canvas_frame.pack(side=TOP, anchor=W)

        self.db_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.db_value_drop.pack(side=LEFT, padx=10, pady=7, anchor=W)

        # Настройка фрейма для последовательной системы

        self.cons_label.pack(side=TOP, padx=10, pady=7, anchor=W)
        self.method_label.pack(side=LEFT, anchor=W, padx=10, pady=7)
        self.method_drop.pack(side=LEFT, anchor=W, padx=10, pady=7)

        self.param_label.pack(side=LEFT, anchor=W, padx=10, pady=7)
        self.p_entry.pack(side=LEFT, anchor=W, padx=10, pady=7)

        self.from_templ_label.pack(side=LEFT, anchor=W, padx=10, pady=7)
        self.from_template_entry.pack(side=LEFT, anchor=W, padx=10, pady=7)

        self.to_templ_label.pack(side=LEFT, anchor=W, padx=10, pady=7)
        self.to_template_entry.pack(side=LEFT, anchor=W, padx=10, pady=7)

        self.run_but.pack(side=LEFT, anchor=W, padx=10, pady=7)

        self.score_label.pack(side=LEFT, anchor=W, padx=10)
        self.score_result.pack(side=LEFT, anchor=W)

        self.frame1.pack(side=TOP, anchor=W, fill="x")
        self.frame2.pack(side=TOP, anchor=W, fill="x")
        self.frame3.pack(side=TOP, anchor=W, fill="x")

        # Настройка фрейма для параллельной системы
        self.parallel_label.pack(side=TOP, padx=10, pady=7, anchor=W)
        self.hist_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.hist_entry.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.scale_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.scale_entry.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.gradient_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.gradient_entry.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dft_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dft_entry.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dct_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dct_entry.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.templ_num_label.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.templ_num_entry.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.parallel_button.pack(side=TOP, padx=10, pady=7, anchor=W)

        self.frame4.pack(side=TOP, anchor=W, fill="x")
        self.frame5.pack(side=TOP, anchor=W, fill="x")
        self.frame6.pack(side=TOP, anchor=W, fill="x")
        self.frame7.pack(side=TOP, anchor=W, fill="x")

        # Настройка первого фрейма
        self.cons_label2.pack(side=TOP, padx=10, pady=7, anchor=W)
        self.method_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.method_drop2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.temp_to_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.temp_to_entry2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.run_but2.pack(side=LEFT, padx=10, pady=7, anchor=W)

        self.parallel_label2.pack(side=TOP, padx=10, pady=7, anchor=W)
        self.hist_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.hist_entry2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.scale_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.scale_entry2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.gradient_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.gradient_entry2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dft_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dft_entry2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dct_label2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.dct_entry2.pack(side=LEFT, padx=10, pady=7, anchor=W)
        self.parallel_button2.pack(side=TOP, padx=10, pady=7, anchor=W)

        # self.scroll_x.pack(side=BOTTOM, fill=X)
        # self.canvas.pack(fill=BOTH, expand=True)

        self.frame8.pack(side=TOP, anchor=W, fill="x")
        self.frame9.pack(side=TOP, anchor=W, fill="x")
        self.frame10.pack(side=TOP, anchor=W, fill="x")

        # Настройка фрейма с Canvas
        self.canvas.pack(side=TOP)

        

    def consistent_experiment(self) -> None:
        """
        Проведение эксперимента с последовательной системой
        и отображение результатов.
        """
        self.canvas.delete("all")
        score, images, templates = recognition(
            self.db_value.get(),
            self.method.get(),
            int(self.p_entry.get()),
            int(self.from_template_entry.get()),
            int(self.to_template_entry.get()),
        )
        self.score_result.config(text=score)

        templ_posx = 300
        templ_posy = 110

        res_posx = 50
        res_posy = 110

        random_indexes = [random.randrange(len(images)) for _ in range(10)]

        self.canvas.create_text(50, 50, text="Test image", fill="black", font="Arial 15")
        self.canvas.create_text(300, 50, text="First image from detected group", fill="black", font='Arial 15')

        for index in random_indexes:
            templ = Image.fromarray(templates[index])
            templ.resize((50, 50))
            templ = ImageTk.PhotoImage(templ)
            self.templates.append(templ)
            self.canvas.create_image(templ_posx, templ_posy, image=templ)

            templ_posy += 80

            img = Image.fromarray(images[index])
            img.resize((50, 50))
            img = ImageTk.PhotoImage(img)
            self.result_images.append(img)
            self.canvas.create_image(res_posx, res_posy, image=img)

            res_posy += 80

    def parallel_experiment(self) -> None:
        """
        Проведение эксперимента с параллельной системой
        и отображение результатов.
        """
        self.canvas.delete("all")
        params = [
            ('hist', int(self.hist_entry.get())),
            ('scale', int(self.scale_entry.get())),
            ('grad', int(self.gradient_entry.get())),
            ('dft', int(self.dft_entry.get())),
            ('dct', int(self.dct_entry.get()))
        ]
        L = int(self.templ_num_entry.get())
        scores = parallel_recognition(
            db_name=self.db_value.get(),
            params=params,
            templ_to=L
        )

        posx = 250
        posy = 250

        image = Image.open(
            DATA_PATH + "results/parallel_experiment_result.png"
        )
        image = image.resize((500, 300))
        image = ImageTk.PhotoImage(image)
        self.result_images.append(image)
        self.canvas.create_image(posx, posy, image=image)


    #######3

    def research(self) -> None:
        """
        Проведение исследований с последовательной системой
        и отображение результатов.
        """
        self.canvas.delete("all")
        best_scores, _, _ = research(self.db_value.get(), self.method2.get(), int(self.temp_to_entry2.get()))
        self.result_images = []

        posx = 250
        posy = 250

        image = Image.open(DATA_PATH + f"results/result_1_n.png")
        image = image.resize((350, 350))
        image = ImageTk.PhotoImage(image)
        self.result_images.append(image)
        self.canvas.create_image(posx, posy, image=image)

        posx += 360

    def parallel_research(self) -> None:
        """
        Проведение исследований с параллельной системой
        и отображение результатов.
        """
        global image
        self.canvas.delete("all")
        self.result_images = []
        params = [
            ('hist', int(self.hist_entry2.get())),
            ('scale', int(self.scale_entry2.get())),
            ('grad', int(self.gradient_entry2.get())),
            ('dft', int(self.dft_entry2.get())),
            ('dct', int(self.dct_entry2.get()))
        ]
        parallel_system_research(self.db_value.get(), params)

        posx = 250
        posy = 250

        image = Image.open(DATA_PATH + "results/parallel_result.png")
        image = image.resize((500, 300))
        image = ImageTk.PhotoImage(image)
        self.result_images.append(image)
        self.canvas.create_image(posx, posy, image=image)
