import cv2
import os
import numpy as np
from IPython import display
from skimage import io
from skimage import color
import shutil
import matplotlib.pyplot as plt

id = 0
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# names related to ids: example ==> Marcelo: id=1,  etc
# names = ['None', 'Giridhar', 'Prabhjot', 'Swetha', 'Kartiki'] 

#reading names from names.txt
names = []
names_file = open('names.txt','r')
names.append('None')
names_text = names_file.read()
for i in names_text.strip().split("\n"):
    person_name = i.strip()
    names.append(person_name)

print(names)

def clearFilesInFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 700) # set video widht
cam.set(4, 700) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('model.yml')


font = cv2.FONT_HERSHEY_SIMPLEX

print("opening camera")

results = {}
lastFewNames = []    
flag = True
predicted_name = ""
unknown_count = 0
msg1 = ""
msg2 = ""
predicted_names = []

while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( gray, scaleFactor = 1.3, minNeighbors = 5, minSize = (int(minW), int(minH)),)

    no_of_faces = len(faces)
    print("No of faces found "+ str(no_of_faces))

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = face_recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 70  and id < len(names)):
            name = names[id]
            confidence = "  {0}%".format(round(100 - confidence*0.3))
            # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            msg1 = "Recognised "+ name
            msg2 = "Attendance marked"
            # if(no_of_faces == len(predicted_names)):
            #     predicted_names.append(name)
            #     flag = False
        else:
            # cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
            name = "unknown"
            unknown_count += 1
            confidence = "  {0}%".format(round(100 - confidence))
            if not os.path.exists("unknown"):
                clearFilesInFolder("unknown")
                shutil.rmtree("unknown")
                os.makedirs("unknown")
            msg1 = "Not Recognised"
            msg2 = "Unknown user"
            cv2.imwrite("unknown/" + str(unknown_count) + ".jpg", gray[y:y+h,x:x+w])
            # cv2.imshow( "unknown", img)
            # cv2.waitKey(5000)
            # if(no_of_faces == len(predicted_names)):
            #     predicted_names.append("unknown")
            #     flag = False

        # print(name , confidence)

        if not os.path.exists("unknown"):
                clearFilesInFolder("unknown")
                shutil.rmtree("unknown")
                os.makedirs("unknown")

        print(results)
        if(name in results):
            results[name] += 1
        else:
            results[name] = 1
            cv2.imwrite("detected_faces/"+ str(name)+ ".jpg", gray[y:y+h,x:x+w])
        
        
        if(no_of_faces == len(predicted_names)):
            predicted_names.append(name)
            flag = False
        else:
            predicted_names.append(name)
            flag = True
        
        # if(len(lastFewNames) < 5):
        #     lastFewNames.append(name)
        # elif(len(lastFewNames) == 5):
        #     if(len(set(lastFewNames)) == 1):
        #         flag = False
        #         predicted_name = lastFewNames[0]
        #         break
        #     else:
        #         del lastFewNames[0]

        # print(lastFewNames)
        print(str(len(predicted_names)))
        # flag = False


        cv2.putText(
                    img, 
                    msg1, 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,0,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    msg2, 
                    (200,500), 
                    font, 
                    1, 
                    (0,0,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    # if(predicted_name):
        # print("Predicted Name : " + predicted_name)
        # print("Attendance of "+predicted_name + " marked successfully.")
    flag = False
    cv2.imshow('camera',img) 
    k = cv2.waitKey(30) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        print(results)
        # print(name)
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# import socket # for socket 
# import sys  
  
# try: 
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
#     print ("Socket successfully created")
# except socket.error as err: 
#     print ("socket creation failed with error" +err )
  
# # default port for socket 
# port = 2001
  
# # try: 
# #     host_ip = socket.gethostbyaddr("10.38.20.27") 
# # except socket.gaierror: 
  
# #     # this means could not resolve the host 
# #     print ("there was an error resolving the host")
# #     sys.exit()
  
# # connecting to the server 
# # s.connect(host_ip, port) 
# s.connect(('127.0.0.1', port)) 

# # if()
# print ("the socket has successfully connected to google" )
# # print(host_ip) 
# for i in predicted_names:
#     s.send(i.encode())
# # s.send("msgserhgs".encode())
# s.close()