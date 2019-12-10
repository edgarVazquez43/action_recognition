"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main(nb_images=10):
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('data/checkpoints/inception.032-1.10.hdf5')

    # Get all our test images.
    images = glob.glob(os.path.join('data', 'test', '**', '*.jpg'))
    images = sorted(images)
    
    print(len(images))

    for i in range(10):
        print(images[i])
    
    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        

        # We take the 50 frames around sample index
        for wide_sample in range(-24, 24):
            print('-'*40)
            image = images[sample + wide_sample]
            
            # Turn the image into an array.
            print(image)
            image_arr = process_image(image, (299, 299, 3))
            image_arr = np.expand_dims(image_arr, axis=0)

            # Predict.
            predictions = model.predict(image_arr)

            # Show how much we think it's each one.
            label_predictions = {}
            for i, label in enumerate(data.classes):
                label_predictions[label] = predictions[0][i]

            sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
                
            for i, class_prediction in enumerate(sorted_lps):
                # Just get the top five.
                if i == 0:
                    title = "  Predict: " + str(class_prediction[0]) + " -  " + str(class_prediction[1])
                if i > 4:
                    break
                print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
                i += 1
                    
            img = mpimg.imread(image)
            plt.title(title)
            plt.imshow(img)
            plt.pause(0.01)
            plt.clf()

if __name__ == '__main__':
    main()






