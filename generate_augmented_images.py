import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

base_folder = "coffee_bean"  
angles = [45, 90, 135, 180, 225, 270]  # Angles of rotation
for root, _, files in os.walk(base_folder):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            input_image_path = os.path.join(root, file)

            try:
                # Load the image
                img = load_img(input_image_path)  # Load image as PIL Image

                # Generate and save rotated images in the same folder
                for angle in angles:
                    rotated_img = img.rotate(angle, expand=True)
                    # Append angle info to the original filename
                    output_filename = f"{os.path.splitext(file)[0]}_rotated_{angle}.jpg"
                    output_path = os.path.join(root, output_filename)
                    rotated_img.save(output_path)
                    print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Error processing {input_image_path}: {e}")
