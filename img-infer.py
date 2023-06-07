import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

my_img = "cat1.jpg"

interpreter = tflite.Interpreter(model_path="cat-dog.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# check the type of the input tensor
floating_model = input_details[0]["dtype"] == np.float32

# NxHxWxC, H:1, W:2
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]
img = Image.open(my_img).resize((width, height))

# add N dim
input_data = np.expand_dims(img, axis=0)

if floating_model:
    input_data = (np.float32(input_data) - 127.5) / 127.5

interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])
