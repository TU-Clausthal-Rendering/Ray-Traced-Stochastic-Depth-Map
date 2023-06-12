# config
dataPath = 'D:/VAO/'

# imports
import numpy as np
import os

# set current directory as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load data
X = np.load(dataPath + '.npy')