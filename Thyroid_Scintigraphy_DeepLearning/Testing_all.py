import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from sklearn.metrics import f1_score
from tensorflow_addons.metrics import F1Score
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


  
print('start')
# Create a Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Ask the user to select an image file
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
imageI = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

temp_dir = r'test'  # Name of the temporary directory
temp_dir2 = r'test\test1'

# Create the temporary directory
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(temp_dir2, exist_ok=True)

image = None  
def onselect(eclick, erelease):
    global image
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    image = imageI[y1:y2, x1:x2]
    plt.close()

fig, ax = plt.subplots()
ax.imshow(imageI, cmap='gray')

selector = RectangleSelector(ax, onselect, useblit=True)
plt.show()
print('remove')
def remove_noise(image):
    axes[0,1].imshow(image, cmap='gray')
    axes[0,1].set_title('Cropped Image')
    axes[0,1].axis('off')  # Turn off axes
    blurred = cv2.GaussianBlur(image, (25, 25), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    #sharpened = cv2.equalizeHist(sharpened)
    axes[1,0].imshow(sharpened, cmap='gray')
    axes[1,0].set_title('De-Noised Image')
    axes[1,0].axis('off')  # Turn off axes
    
    #denoised_image = cv2.equalizeHist(denoised_image1)
    return sharpened




fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0,0].imshow(imageI, cmap='gray')
axes[0,0].set_title('Original Image')
axes[0,0].axis('off')  # Turn off axes


DeNoised_image = remove_noise(image)
cv2.imwrite(r'test\test1\single_image.jpg', DeNoised_image)
# Copy your single image to the temporary directory
#shutil.copy(image_path, os.path.join(temp_dir2, 'single_image.jpg'))
saved_model_path = r'CNN_InceptV3.h5'
loaded_model = load_model(saved_model_path)


# Create the data generator
test_data_generator = ImageDataGenerator(rescale=1./255)

# Set up the data flow
test_data_flow = test_data_generator.flow_from_directory(
    temp_dir,
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
)

# Load the single image using the generator
single_image_batch, single_label_batch = next(test_data_flow)

# Now you can perform predictions using your model
predictions = loaded_model.predict(single_image_batch,verbose=0)
#print(predictions)
predicted_class = np.argmax(predictions)
if predicted_class == 0:
  Pred_Class='Blocked'
elif predicted_class == 1:
  Pred_Class='Cold'
elif predicted_class == 2:
  Pred_Class='Functional'
elif predicted_class == 3:
  Pred_Class='MNG'
elif predicted_class == 4:
  Pred_Class='Normal'
else: Pred_Class='Unknown'
 
print(f" Predicted Class: {Pred_Class}")


image = Image.open(r'test\test1\single_image.jpg').convert('L')

# Initialize the drawing context
draw = ImageDraw.Draw(image)

# Define text
text1= 'Predicted Class : '
text2 = Pred_Class
text = text1+ text2
font_size = 30  # Increase the font size

# Load a font with the specified size
font = ImageFont.truetype(r'font.ttf', font_size)

bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Calculate text position for centering
image_width, image_height = image.size
x = (image_width - text_width) // 2
y = image_height - text_height - 50

# Draw the text on the image with white color
draw.text((x, y), text, font=font, fill=255)



# Convert the PIL image to a numpy array
image_array = np.array(image)

# Display the image using matplotlib

axes[1,1].imshow(image_array, cmap='gray')
axes[1,1].set_title('Predicted Image')
axes[1,1].axis('off')  # Turn off axes

plt.show()



# Clean up: Remove the temporary directory if needed
shutil.rmtree(temp_dir)

