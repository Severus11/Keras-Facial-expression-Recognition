import numpy as np
import csv
from PIL import Image    
import matplotlib.pyplot as plt

counter = dict()

with open('C:\Users\parth\Downloads\test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    # skip headers
    next(csv_reader)

    for row in csv_reader:

        pixels = row[1] # without label
        pixels = np.array(pixels)
        pixels = pixels.reshape((48, 48))

        label = row[0]

        if label not in counter:
            counter[label] = 0
        counter[label] += 1

        filename = '{}{}.png'.format(label, counter[label])
        plt.imsave(filename, pixels)

        print('saved:', filename)