import numpy as np
import os

# combines from 0 to (NUM_FILES - 1)

# set current directory as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
cleanupFiles = []

def combineFiles(baseName):
	filenames = []
	i = 0
	while True:
		fname = f'{baseName}{i}.npy'
		if not os.path.isfile(fname):
			break
		filenames.append(fname)
		cleanupFiles.append(fname)
		i += 1
	
	combined = np.concatenate([np.load(filename) for filename in filenames])
	print("writing", f'{baseName}.npy')
	np.save(f'{baseName}.npy', combined)

combineFiles('bright_')
combineFiles('dark_')
combineFiles('ref_')

combineFiles('depth_')
combineFiles('invDepth_')

# cleanup the old files
for filename in cleanupFiles:
	os.remove(filename)