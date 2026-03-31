from PIL import Image, ImageDraw
import os

# Create a 28x28 black image
img = Image.new('L', (28, 28), color=0)
draw = ImageDraw.Draw(img)

# Draw a rough, thick white line representing a "1"
draw.line((14, 4, 14, 24), fill=255, width=3)

# Save it to the models directory
img.save('../models/test_one.png')
print("Saved test_one.png!")