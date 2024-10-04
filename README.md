# Avataar_Assignment
The task is to write an executable code that takes the location of the image and the text prompt from the command line argument and outputs a generated image.

**#Execution of code**

first, we need to set up the environment.
Install all the necessary Python packages 

>> pip install torch diffusers pillow opencv-python argparse

The input image needs to be in a valid format (jpeg/jpg)
the text prompt should describe the scene in a clear manner that would be used to create the background of the image.
to run the code take the following syntax as an example //write in the terminal:

>> python run.py --image ./examples/example1.jpg --text-prompt "Product in a kitchen placed on a sleek kitchen island with a dish on the left side, high resolution, kitchen has bright and warm lighting" --output ./generated.png

**#Approach**
The task was to generate an image with the object image placed into the scene generated from a text prompt. The goal was to create a natural-looking output, and object image should be unaltered and the output should align with the text prompt.

to achieve this I used Stable Diffusion as it is one of the most advanced generative models for text-to-image tasks.

I first started with StableDiffusionInpaintPipeline since this model allows replacing parts of an image, but it was not able to generate a background solely on text. I was getting an error that the image was not in the correct format. 

I then switched to StableDiffusionPipeline which was able to generate the background image based on the text prompt given making it suitable for the task. 

I used openCV and Pillow (PIL) to create mask from the object image. Masking was necessary to remove the white background and to ensure proper transparency during the placement of the object image in the generated background.

I tried to enhance the realism of the image using pillow library, to adjust the brightness and contrast of the image.

After generating a background using the StableDiffusionPipeline from the text prompt I combine the object image and background image using mask. the object image size was adjusted so that it could fit naturally in the scene.

the final output is saved on the path defined by the user.
