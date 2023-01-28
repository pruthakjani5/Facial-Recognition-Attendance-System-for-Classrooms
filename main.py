import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
import gooeypie as gp
# import openpyxl
from PIL import ImageGrab

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

encodeList=[]
def findEncodings(images):
    # encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

nameList=[]

def markAttendance(name):
    #book=openpyxl.load_workbook('C:\\Users\\Ansh\\Desktop\\phyproject\\Attendance.xlsx')
    with open('Attendance.csv', 'a') as f:
        # myDataList = f.readlines()
        # nameList = []
        # for line in myList:
            # entry = line
            # nameList.append(entry[0])
            # if name in nameList:
            csvwriter = csv.writer(f, delimiter=" ")
            # csvwriter.writerow(name)
            now = datetime.now()
            dtString = now.strftime('%D:%H:%M:%S')
            csvwriter.writerow(f'{name} {dtString}')
    print(name)
    # print(myDataList)
# def markAttendance(name):
    # with open('C:\\Users\\Ansh\\Desktop\\phyproject\\Attendance.csv', 'r+') as f:
    #     myDataList = f.readlines()
    #     for line in myDataList:
    #         entry = line.split(',')
    #         today = datetime.now().strftime("%d/%m/%Y")

    #         if name in line and entry[1] == today:
    #             print(entry[1] + entry[2])

    #         elif name not in line and entry[1] == today:
    #             now = datetime.now().strftime("%d/%m/%Y,%H:%M")
    #             f.write(f'\n{name},{now}')
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#             if name not in nameList:
#                 now = datetime.now()
#                 dtString = now.strftime('%H:%M:%S')
#                 f.writelines(f'n{name},{dtString}')
### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
# cap.release()
while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0))
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            app = gp.GooeyPieApp('FRASC')
            app.width = 250 
            app.height =100
            btn = gp.Button(app, 'Mark Attendance', markAttendance(name))
            # lbl = gp.Label(app, '')
            app.set_grid(2, 1)
            app.add(btn, 1, 1, align='center')
            # app.add(lbl, 1, 2, align='center')
            app.run()
            # markAttendance(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    # print(name)