from PIL import Image

import os

NUM_OF_IMAGES = 120

data_dir = "./images/"
out_dir = "./gif/"

gif_name = "decision_region_depending_on_k-knn.gif"

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

images = []

for i in range(NUM_OF_IMAGES):
    path = os.path.join(data_dir, "decision_region-k=" + str(i+1) + ".png")
    img = Image.open(path)
    images.append(img)

try:
    images[0].save(os.path.join(out_dir, gif_name), save_all=True, append_images=images[1:], optimize=False, duration=400, loop=1)

except:
    pass