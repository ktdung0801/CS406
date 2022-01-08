from tkinter import Toplevel, Label, Scale, Button, messagebox, HORIZONTAL, LEFT, RIGHT
import cv2


class SettingFrame(Toplevel):
    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.title("Setting")
        w = 300 # width for the Tk root
        h = 200 # height for the Tk root
        # get screen width and height
        ws = self.winfo_screenwidth() # width of the screen
        hs = self.winfo_screenheight() # height of the screen
        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        # set the dimensions of the screen 
        # and where it is placed
        self.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.resizable(0,0)

        self.blocks_label = Label(self, text="Number of pixel blocks")
        self.blocks_scale = Scale(self, from_=1, to_=30, length=250, resolution=1, orient=HORIZONTAL)
        self.factor_label = Label(self, text="Factor")
        self.factor_scale = Scale(self, from_=1, to_=30, length=250, resolution=1, orient=HORIZONTAL)
        self.save_button = Button(self, text="Save", width=10)
        self.cancel_button = Button(self, text="Cancel", width=10)

        self.blocks_scale.set(self.master.blocks)
        self.factor_scale.set(self.master.factor)

        self.save_button.bind("<ButtonRelease>", self.save_button_released)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)

        self.blocks_label.pack()
        self.blocks_scale.pack()
        self.factor_label.pack()
        self.factor_scale.pack()
        self.cancel_button.pack(side=RIGHT)
        self.save_button.pack(side=RIGHT)
        
    def save_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.save_button:
            self.master.blocks = self.blocks_scale.get()
            self.master.factor = self.factor_scale.get()
            messagebox.showinfo(title=None, message="Lưu thành công!")

    def cancel_button_released(self, event):
        if self.winfo_containing(event.x_root, event.y_root) == self.cancel_button:
            self.destroy()
