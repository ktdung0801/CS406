import tkinter as tk
from tkinter import ttk
from editBar import EditBar
from imageViewer import ImageViewer
import os
import cv2

class Main(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector",
                                        "res10_300x300_ssd_iter_140000.caffemodel"])
        self.net = cv2.dnn.readNet(prototxtPath, weightsPath)

        self.filename = ""
        self.original_image = None
        self.processed_image = None
        self.is_image_selected = False
        self.is_crop_state = False
        self.blocks = 20
        self.factor = 3

        self.title("Image Blur")
        #self.call('wm', 'iconphoto', self._w, tk.PhotoImage(file='uit.png'))
        self.iconphoto(False, tk.PhotoImage(file='icon.png'))

        # get screen width and height
        w = self.winfo_screenwidth() - 100 # width of the screen
        h = self.winfo_screenheight() - 100 # height of the screen
        # calculate x and y coordinates for the Tk root window
        x = (w/2) - (w/2)
        y = (h/2) - (h/2)
        self.geometry('%dx%d+%d+%d' % (w, h, x+40, y))

        self.editbar = EditBar(master=self)
        separator1 = ttk.Separator(master=self, orient=tk.HORIZONTAL)
        self.image_viewer = ImageViewer(master=self)

        self.setting_frame = None

        self.editbar.pack(pady=10)
        separator1.pack(fill=tk.X, padx=20, pady=5)
        self.image_viewer.pack(fill=tk.BOTH, padx=20, pady=10, expand=1)