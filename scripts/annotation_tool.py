import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None


class AnnotationTool:
    def __init__(self, root, image_dir, annotation_dir):
        self.root = root
        self.root.title("Layout Annotation Tool")
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        os.makedirs(annotation_dir, exist_ok=True)
        self.image_paths = []
        self.get_image_paths(image_dir)
        self.current_image_idx = 0

        self.setup_ui()

        if self.image_paths:
            self.load_image(self.image_paths[0])
        
    def get_image_paths(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
    
    def setup_ui(self):
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X)
 
        prev_btn = tk.Button(btn_frame, text="Previous", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        next_btn = tk.Button(btn_frame, text="Next", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        save_btn = tk.Button(btn_frame, text="Save", command=self.save_annotation)
        save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        #Drawing mode
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.rectangle = None
        self.boxes = []
        
        #Canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
    
    def load_image(self, path):
        self.current_image_path = path
        image = Image.open(path)

        max_size = 800
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.boxes = []

        rel_path = os.path.relpath(path, self.image_dir)
        doc_name = os.path.dirname(rel_path)
        file_name = os.path.splitext(os.path.basename(rel_path))[0]
        
        annotation_doc_dir = os.path.join(self.annotation_dir, doc_name)
        os.makedirs(annotation_doc_dir, exist_ok=True)
        
        annotation_path = os.path.join(annotation_doc_dir, f"{file_name}.json")
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                self.boxes = data.get('text_regions', [])

            for box in self.boxes:
                self.canvas.create_rectangle(box[0], box[1], box[2], box[3], outline='red', width=2)
    
    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rectangle = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )
    
    def on_mouse_move(self, event):
        if self.drawing:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rectangle, self.start_x, self.start_y, cur_x, cur_y)
    
    def on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)

            x1 = min(self.start_x, end_x)
            y1 = min(self.start_y, end_y)
            x2 = max(self.start_x, end_x)
            y2 = max(self.start_y, end_y)

            self.boxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    def save_annotation(self):
        rel_path = os.path.relpath(self.current_image_path, self.image_dir)
        doc_name = os.path.dirname(rel_path)
        file_name = os.path.splitext(os.path.basename(rel_path))[0]
        
        annotation_doc_dir = os.path.join(self.annotation_dir, doc_name)
        os.makedirs(annotation_doc_dir, exist_ok=True)
        
        annotation_path = os.path.join(annotation_doc_dir, f"{file_name}.json")
        
        data = {"text_regions": self.boxes}
        
        with open(annotation_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Saved annotations to {annotation_path}")
    
    def next_image(self):
        if self.current_image_idx < len(self.image_paths) - 1:
            self.current_image_idx += 1
            self.load_image(self.image_paths[self.current_image_idx])
    
    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_image(self.image_paths[self.current_image_idx])

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root, r"data\processed\images", r"data\processed\annotations")
    root.mainloop()
