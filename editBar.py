from tkinter import *
from tkinter.ttk import Frame, Button
import cv2
from tkinter import filedialog
from face_blurring.face_blurring import anonymize_face_simple
from face_blurring.face_blurring import anonymize_face_pixelate
import os
import numpy as np
from settingFrame import SettingFrame

class EditBar(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master=master)

        self.new_button = Button(self, text="Open")
        self.save_as_button = Button(self, text="Save As")
        self.blurAll_button = Button(self, text="Blur All")
        self.pixelateAll_button = Button(self, text="Pixelate All")
        self.crop_button = Button(self, text="Manual Blurring")
        self.pixelate_button = Button(self, text="Manual Pixelation")
        self.clear_button = Button(self, text="Revert")
        self.setting_button = Button(self, text="Setting")

        self.new_button.bind("<ButtonRelease>", self.new_button_released)
        self.save_as_button.bind("<ButtonRelease>", self.save_as_button_released)
        self.blurAll_button.bind("<ButtonRelease>", self.blurAll_button_released)
        self.pixelateAll_button.bind("<ButtonRelease>", self.pixelateAll_button_released)
        self.crop_button.bind("<ButtonRelease>", self.crop_button_released)
        self.pixelate_button.bind("<ButtonRelease>", self.pixelate_button_released)
        self.clear_button.bind("<ButtonRelease>", self.clear_button_released)
        self.setting_button.bind("<ButtonRelease>", self.setting_button_released)

        self.new_button.pack(side=LEFT)
        self.save_as_button.pack(side=LEFT)
        self.setting_button.pack(side=LEFT)
        self.blurAll_button.pack(side=LEFT)
        self.pixelateAll_button.pack(side=LEFT)
        self.crop_button.pack(side=LEFT)
        self.pixelate_button.pack(side=LEFT)
        self.clear_button.pack(side=LEFT)

    def new_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.new_button:
            if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

            filename = filedialog.askopenfilename()
            image = cv2.imread(filename)

            if image is not None:
                self.master.filename = filename
                self.master.original_image = image.copy()
                self.master.processed_image = image.copy()
                self.master.image_viewer.show_image()
                self.master.is_image_selected = True

    def save_as_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_as_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()

                original_file_type = self.master.filename.split('.')[-1]
                filename = filedialog.asksaveasfilename()
                filename = filename + "." + original_file_type

                save_image = self.master.processed_image
                cv2.imwrite(filename, save_image)

                self.master.filename = filename

    def blurAll_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.blurAll_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                self.master.processed_image = self.process_face(method = "blur")
                self.master.image_viewer.show_image()

    def pixelateAll_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.pixelateAll_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                self.master.processed_image = self.process_face(method = "pixelate")
                self.master.image_viewer.show_image()

    def crop_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.crop_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                else:
                    self.master.image_viewer.blur = 1
                    self.master.image_viewer.activate_crop()

    def pixelate_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.pixelate_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                else:
                    self.master.image_viewer.pixelate = 1
                    self.master.image_viewer.activate_crop()

    def clear_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.clear_button:
            if self.master.is_image_selected:
                if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
                self.master.processed_image = self.master.original_image.copy()
                self.master.image_viewer.show_image()

    def process_face(self, method):
        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        image = self.master.processed_image.copy()
        (h, w) = image.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.master.net.setInput(blob)
        detections = self.master.net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is greater
            # than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = image[startY:endY, startX:endX]

                # check to see if we are applying the "simple" face blurring method
                if method == "blur":
                    face = anonymize_face_simple(face, factor=self.master.factor)

                # otherwise, we must be applying the "pixelated" face
                # anonymization method
                elif method == "pixelate":
                    face = anonymize_face_pixelate(face, blocks=self.master.blocks)

                # store the blurred face in the output image
                image[startY:endY, startX:endX] = face
        
        return image
        #end

    def setting_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.setting_button:
            if self.master.is_crop_state:
                    self.master.image_viewer.deactivate_crop()
            
            self.master.adjust_frame = SettingFrame(master=self.master)
            self.master.adjust_frame.grab_set()

