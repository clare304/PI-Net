import os
from PIL import Image
from pathlib import Path

def convert_images_to_nnunet_format(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(input_dir, "nnunet_converted")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif')
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if os.path.isdir(file_path):
            continue
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in supported_extensions:
            continue
        
        file_base = os.path.splitext(filename)[0]
        new_filename = f"{file_base}_0000.png"
        new_file_path = os.path.join(output_dir, new_filename)
        
        try:
            with Image.open(file_path) as img:
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(new_file_path, 'PNG')
            print(f"Successfully converted: {filename} -> {new_filename}")
        
        except Exception as e:
            print(f"Failed to convert {filename}: {str(e)}")

if __name__ == "__main__":
    INPUT_FOLDER = r"your_folder"       #Replace with your own path
    convert_images_to_nnunet_format(INPUT_FOLDER)