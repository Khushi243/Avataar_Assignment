import cv2
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
import argparse

# Loading the Stable Diffusion text-to-image model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu" #GPU or CPU
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    return model

# masking input image 
def create_object_mask(image):
    grayscale_img = ImageOps.grayscale(image)
    np_img = np.array(grayscale_img)
    _, mask = cv2.threshold(np_img, 240, 255, cv2.THRESH_BINARY_INV)

    # Creating a PIL Image from the mask
    mask_image = Image.fromarray(mask).convert("L")
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(4))
    
    return mask_image

# Adjusting lighting to match the background 
def adjust_lighting(image, background):
    #object image 
    object_brightness = ImageEnhance.Brightness(image)
    object_contrast = ImageEnhance.Contrast(image)

    #background image
    bg_brightness = ImageEnhance.Brightness(background)
    bg_contrast = ImageEnhance.Contrast(background)

    # average pixel intensity for brightness
    avg_brightness_obj = np.mean(np.array(object_brightness.enhance(1)))
    avg_brightness_bg = np.mean(np.array(bg_brightness.enhance(1)))

    #contrast
    avg_contrast_obj = np.std(np.array(object_contrast.enhance(1)))
    avg_contrast_bg = np.std(np.array(bg_contrast.enhance(1)))

    # Adjusting object brightness and contrast to match background
    if avg_brightness_obj != 0:
        brightness_factor = avg_brightness_bg / avg_brightness_obj
        image = object_brightness.enhance(brightness_factor)

    if avg_contrast_obj != 0:
        contrast_factor = avg_contrast_bg / avg_contrast_obj
        image = object_contrast.enhance(contrast_factor)

    return image

# Generateing the background scene using the text prompt
def generate_background(model, prompt):
    generated_img = model(prompt=prompt).images[0]
    return generated_img

# Combining the object with the generated background 
def combine_image_object_with_bg(image, mask, background):

    adjusted_object = adjust_lighting(image, background)

    # Resizing object and mask to match the size of the background
    adjusted_object = adjusted_object.resize(background.size, Image.LANCZOS)
    mask = mask.resize(background.size, Image.LANCZOS)

    adjusted_object = adjusted_object.convert("RGBA")

    #Ensure object is being pasted
    print(f"Adjusted Object Size: {adjusted_object.size}")
    print(f"Generated Background Size: {background.size}")
    print(f"Mask Size: {mask.size}")

    # Placing the object in the center of the background
    position = ((background.width - adjusted_object.width) // 2,
                (background.height - adjusted_object.height) // 2)

    # Pasting the object back into the scene using the mask for transparency
    background.paste(adjusted_object, position, mask=mask)

    return background

#image generation and object insertion
def process_image(model, image_path, prompt, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Creating mask for the object
    mask = create_object_mask(img)

    # Generating background from text prompt
    generated_bg = generate_background(model, prompt)

    # Combining object and background
    final_image = combine_image_object_with_bg(img, mask, generated_bg)

    # Saving the final output image
    final_image.save(output_path)
    print(f"Generated image saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate image with object placed naturally in a scene")
    parser.add_argument("--image", type=str, required=True, help="Path to the input object image")
    parser.add_argument("--text-prompt", type=str, required=True, help="Text prompt describing the scene")
    parser.add_argument("--output", type=str, required=True, help="Path to save the generated image")
    args = parser.parse_args()

    model = load_model()

    process_image(model, args.image, args.text_prompt, args.output)

if __name__ == "__main__":
    main()
