import numpy as np
import cv2
import os
'''
emotion_list= ['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise','Neutral']
for i in range(0,7):
    x =emotion_list[i]
    os.mkdir(x)

'''
os.mkdir("Test_img")
with open('test.csv') as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size

for i in range(0,num_of_instances):
    try:
        #emotion, img = lines[i].split(",")
        img = lines[i]
        img = img.replace('"', '')
        img = img.replace('\n', '')
        pixels = img.split(" ")

        pixels = np.array(pixels, 'int32')
        image = pixels.reshape(48, 48)

        #x = int(emotion)
        #path_file_name = f"{emotion_list[x]}/{i}_{emotion}.jpg"
        path_file_name=f"Test_img/Test_Img_{i}.jpg"
        cv2.imwrite(path_file_name, image)

    except Exception as ex:
        print(ex)

