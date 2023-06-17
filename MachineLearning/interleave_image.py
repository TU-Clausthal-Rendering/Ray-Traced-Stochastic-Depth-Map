import sys
import numpy as np

# get filename from command line argument
filename = sys.argv[1]

img = np.load(filename)
original_shape = img.shape

# create numpy array with 4*shape
new_shape = (original_shape[1]*4, original_shape[2]*4)

# create new numpy array with new shape
new_img = np.zeros(new_shape, dtype=np.float32)

for layer in range(original_shape[0]):
	offsetX = layer % 4
	offsetY = layer // 4
	for x in range(original_shape[1]):
		xQuad = x*4 + offsetX
		for y in range(original_shape[2]):
			yQuad = y*4 + offsetY
			# copy pixel to new image
			new_img[xQuad, yQuad] = img[0, x, y, 0]

# save new image
# remove .npy from filename
filename = filename[:-4]
np.save(f'{filename}_interleaved.npy', new_img)