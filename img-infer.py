import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import subprocess


def resize_and_rescale_image(image_path, input_shape):
    # Load the image using PIL
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize((input_shape[2], input_shape[1]))

    # Convert the image to match what the model was trained on
    input_data = np.array(resized_image, dtype=np.float32)
    input_data = (input_data / 127.5) - 1.0

    # Set the batch size to 1, since it's a single image
    input_data = np.expand_dims(input_data, axis=0)

    # # assumes the input tensor wants float32
    # # can check/toggle this with
    # # floating_model = input_details[0]["dtype"] == np.float32
    # # if floating_model:
    # input_data = (np.float32(input_data) - 127.5) / 127.5

    return input_data


def picam_snap(image_path):
    # Capture the image using libcamera and save it to the specified path
    subprocess.run(["libcamera-still", "-o", image_path])


def tflite_infer(interpreter, input_data):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return np.squeeze(output_data)


# Create the tflite Interpreter
model_path = "cat-dog.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Take a picture
image_path = "mypic.jpg"
picam_snap(image_path)
resized_image = resize_and_rescale_image(image_path, input_details[0]["shape"])

# Conduct inference
infer_result = tflite_infer(interpreter, resized_image)

print("Inference result:", infer_result)
