"""
Classify a few images through our CNN.
"""
import cv2
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

    # Load the model
    print(" --- >   Loading model...")
    model = load_model('data/checkpoints/inception.032-1.10.hdf5')

    # Open camera
    print(" --- >   Open camera...")
    cap = cv2.VideoCapture(2)
    cap.set(3, 640)
    cap.set(4, 480)
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Turn the image into an array.
        img2= cv2.resize(frame, dsize=(299,299), interpolation = cv2.INTER_CUBIC)
        #Numpy array
        np_image_data = np.asarray(img2)
        #maybe insert float convertion here - see edit remark!
        np_final = np.expand_dims(np_image_data,axis=0)

        
        # Predict.
        predictions = model.predict(np_final)

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
            #print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

        print(title)
        # Display the resulting frame
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    

    
    # for _ in range(nb_images):
    #     print('-'*80)
    #     # Get a random row.
    #     sample = random.randint(0, len(images) - 1)
        

    #     # We take the 50 frames around sample index
    #     for wide_sample in range(-24, 24):
    #         print('-'*40)
    #         image = images[sample + wide_sample]
            
    #         # Turn the image into an array.
    #         print(image)
    #         image_arr = process_image(image, (299, 299, 3))
    #         image_arr = np.expand_dims(image_arr, axis=0)

    #         # Predict.
    #         predictions = model.predict(image_arr)

    #         # Show how much we think it's each one.
    #         label_predictions = {}
    #         for i, label in enumerate(data.classes):
    #             label_predictions[label] = predictions[0][i]

    #         sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
                
    #         for i, class_prediction in enumerate(sorted_lps):
    #             # Just get the top five.
    #             if i == 0:
    #                 title = "  Predict: " + str(class_prediction[0]) + " -  " + str(class_prediction[1])
    #             if i > 4:
    #                 break
    #             print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
    #             i += 1
                    
    #         img = mpimg.imread(image)
    #         plt.title(title)
    #         plt.imshow(img)
    #         plt.pause(0.01)
    #         plt.clf()

if __name__ == '__main__':
    main()






