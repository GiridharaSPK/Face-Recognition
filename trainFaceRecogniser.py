import cv2
import os
import numpy as np
from IPython import display
from skimage import io
from skimage import color
import matplotlib.pyplot as plt


def get_detected_faces(cascade, test_image, scaleFactor, minNeighbours):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors= minNeighbours)
    print("No. of faces found : " , len(faces_rect))
    faces = []
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
        face = gray_image[y:y+h, x:x+w]
        faces.append(face)

    return image_copy, faces

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def prepare_training_data_from_dataset():
    print("Preparing train data from dataset")
    faces = []
    labels = []
    count = 0

    
    for i in os.listdir('dataset'): #i is Giridhar folder
        count = count +1
        print("label "+ str(count) + " for " + i)
        names_text_file = open("names.txt","a")
        names_text_file.write(i+'\n')
        for j in os.listdir('dataset\\'+i): #j is image in Giridhar folder
            temp_face = cv2.imread('dataset/'+i+'/'+j, cv2.IMREAD_GRAYSCALE)
            label = count
            # full, face = get_detected_faces(haar_cascade_face, temp_face, 1.2, 5)
            # if(len(face)):
                # faces.append(face[0])
                # labels.append(label)
            faces.append(temp_face)
            labels.append(label)
        names_text_file.close()
    return [faces, labels]

# os.remove("names.txt")
open('names.txt', 'w').close()
[faces, labels] = prepare_training_data_from_dataset()
print( "Faces :" +  str(len(faces)) + "   Labels :" + str(len(labels)))

print("Dataset prepared")
# #TRAINING
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Training dataset")
face_recognizer.train(faces, np.array(labels))

print("Training complete")
face_recognizer.write('model.yml') 

# def get_giridhar_data():
#     print("Preparing training dataset of Giridhar")
#     faces = []
#     labels = []
#     for i in os.listdir("dataset\Giridhar"):
#         img = io.imread('dataset/Giridhar/'+i)
#         label = 1
#         full, face = get_detected_faces(haar_cascade_face, img, 1.2, 5)
#         # io.imshow(faces[0])
#     #     io.imshow(full)
#     #     plt.figure()
#     #     plt.imshow(full)
# #         giri = face[0]
#         if(len(face)):
#             giridhar = face[0]
#     #         plt.figure()
#     #         plt.imshow(face[0])
#             faces.append(face[0])
#             labels.append(label)
# #     test_im = io.imread('Giridhar.jpg')
# #     test_im = giridhar
#     return [faces, labels, giridhar]


# def get_prabhjot_data():
#     print("Preparing training dataset of Prabhjot")
#     faces = []
#     labels = []
#     for i in os.listdir("dataset\Prabhjot"):
#         img = io.imread('dataset/Prabhjot/'+i)
#         label = 2
#         full, face = get_detected_faces(haar_cascade_face, img, 1.2, 7)
#         # io.imshow(faces[0])
#     #     io.imshow(full)
#     #     plt.figure()
#     #     plt.imshow(full)
#         if(len(face)):
#             prab = face[0]
#     #         plt.figure()
#     #         plt.imshow(face[0])
#             faces.append(face[0])
#             labels.append(label)
#     test_im = prab
#     return  [faces, labels, test_im]

# def get_swetha_data():
#     print("Preparing training dataset of Swetha")
#     faces = []
#     labels = []
#     for i in os.listdir("dataset\Swetha"):
#         img = io.imread('dataset/Swetha/'+i)
#         label = 2
#         full, face = get_detected_faces(haar_cascade_face, img, 1.2, 7)
#         # io.imshow(faces[0])
#     #     io.imshow(full)
#     #     plt.figure()
#     #     plt.imshow(full)
#         if(len(face)):
#             swetha = face[0]
#     #         plt.figure()
#     #         plt.imshow(face[0])
#             faces.append(face[0])
#             labels.append(label)
#     test_im = swetha
#     return  [faces, labels, test_im]


# # def get_swetha_data():
# #     print("Preparing training dataset of swetha")
# #     faces = []
# #     labels = []
# #     img = io.imread('dataset/Swetha.jpg')
# #     label = 3
# #     full, face = get_detected_faces(haar_cascade_face, img, 1.2, 4)
# #     swe = face[0]
# #     for i in face:
# # #     plt.figure()
# # #     plt.imshow(i)
# #         faces.append(i)
# #         labels.append(label)
# #     test_im = swe
# #     return [faces, labels, test_im]

# def get_kartiki_data():
#     print("Preparing training dataset of Kartiki")
#     faces = []
#     labels = []
#     for i in os.listdir("training-data\Kartiki"):
#         img = io.imread('training-data/Kartiki/'+i)
#         label = 4
#         full, face = get_detected_faces(haar_cascade_face, img, 1.2, 5)
#         # io.imshow(faces[0])
#     #     io.imshow(full)
#     #     plt.figure()
#     #     plt.imshow(full)
# #         giri = face[0]
#         if(len(face)):
#             kartiki = face[0]
#     #         plt.figure()
#     #         plt.imshow(face[0])
#             faces.append(face[0])
#             labels.append(label)
#     return [faces, labels, kartiki]

# def prepare_training_dataset():
#     print("Preparing training dataset")
#     faces = []
#     labels = []
#     faces = faces + get_giridhar_data()[0] + get_prabhjot_data()[0] + get_swetha_data()[0] + get_kartiki_data()[0] #+ get_aamir_data()[0] + get_amitabh_data()[0] + get_akshay_data()[0]
#     labels = labels + get_giridhar_data()[1]  + get_prabhjot_data()[1] + get_swetha_data()[1] + get_kartiki_data()[1] #+ get_aamir_data()[1] + get_amitabh_data()[1] + get_akshay_data()[1]
#     testfaces = []
#     testfaces.append(get_giridhar_data()[2])
#     testfaces.append(get_prabhjot_data()[2])
#     testfaces.append(get_swetha_data()[2])
#     testfaces.append(get_kartiki_data()[2])
#     return faces, labels, testfaces

# #PREPARE DATA
# faces, labels, testfaces = prepare_training_dataset()
# print("Dataset prepared")
# #TRAINING
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# print("Training dataset")
# face_recognizer.train(faces, np.array(labels))

# print("Training complete")
# face_recognizer.write('model.yml') 