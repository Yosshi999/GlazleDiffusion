banner = """
 _____ _           _       ______ _  __  __           _             
|  __ \ |         | |      |  _  (_)/ _|/ _|         (_)            
| |  \/ | __ _ ___| | ___  | | | |_| |_| |_ _   _ ___ _  ___  _ __  
| | __| |/ _` |_  / |/ _ \ | | | | |  _|  _| | | / __| |/ _ \| '_ \ 
| |_\ \ | (_| |/ /| |  __/ | |/ /| | | | | | |_| \__ \ | (_) | | | |
 \____/_|\__,_/___|_|\___| |___/ |_|_| |_|  \__,_|___/_|\___/|_| |_|
                                                                    
Copyright (C) 2024 Yosshi999

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
print(banner)

import threading
import time
import sys
import os
import tkinter
from PIL import ImageTk, Image, ImageDraw
home = os.path.expanduser("~")
glaze = os.path.join(home, ".glaze", "base", "base")
os.environ["TCL_LIBRARY"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-packages/tcl/tcl8.6")
from diffusers import StableDiffusionPipeline
import torch

print("loading library...")
pipe = StableDiffusionPipeline.from_pretrained(glaze, torch_dtype=torch.float16)
mode = "CPU"
if torch.cuda.is_available():
    print("CUDA available")
    pipe.to("cuda")
    mode = "GPU"

print("opening window...")
root = tkinter.Tk()
root.title(f"Glazle Diffusion ({mode} mode)")
root.geometry("600x600")
root.pipe = pipe
canvas = tkinter.Canvas(bg="black", width=512, height=512)
image = Image.new("RGB", (512, 512))
root.pim = ImageTk.PhotoImage(image)
root.cim = canvas.create_image(0, 0, anchor="nw", image=root.pim)
canvas.pack()
txt = tkinter.Entry(width=512)
txt.insert(0, "a photo of an astronaut riding a horse on mars")
txt.pack()

root.stop_now = threading.Event()
def change_img(image):
    root.pim = ImageTk.PhotoImage(image)
    canvas.itemconfig(root.cim, image=root.pim)

def t_progress(_pipe, step, _timestep, _kwargs):
    if root.stop_now.is_set():
        print("stopping")
        raise
    image = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(image)
    draw.text((256, 256), f"Please wait... {step/50:.1%}", "white")
    root.after(100, change_img, image)
    return {}

def t_finish(image):
    change_img(image)
    root.button.configure(state=tkinter.ACTIVE)

def t_generate(prompt):
    image = root.pipe(prompt, callback_on_step_end=t_progress).images[0]
    root.after(100, t_finish, image)

root.thread = None
def start_generate():
    root.button.configure(state=tkinter.DISABLED)
    image = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(image)
    draw.text((256, 256), "Please wait...", "white")
    change_img(image)
    root.thread = threading.Thread(target=t_generate, args=[txt.get()], daemon=True)
    root.thread.start()

def close():
    if root.thread is not None:
        root.stop_now.set()
    root.destroy()

button = tkinter.Button(text="Generate!", command=start_generate)
root.button = button
button.pack()
root.protocol("WM_DELETE_WINDOW", close)

root.mainloop()