import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
import openpyxl
from PIL import ImageGrab
import gooeypie as gp 
import sys 
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tkinter.font as tkFont
import shutil 
import imutils

path = 'Training_images'

def add_images():
    directory = filedialog.askdirectory()
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            shutil.copy(image_path, path)

class App:
    def __init__(self, root):
        #setting title
        root.title("FRASC")
        #setting window size
        root.geometry("575x415")
        root.resizable(width=True, height=True)
        root.config(bg="#98ff98")

        GButton1=tk.Button(root)
        GButton1["bg"] = "#f23232"
        ft = tkFont.Font(family='Times',size=10)
        GButton1["font"] = ft
        GButton1["fg"] = "#ffffff"
        GButton1["justify"] = "center"
        GButton1["text"] = "Add Images"
        GButton1.place(x=130,y=150,width=70,height=25)
        GButton1["command"] = add_images

        GButton2=tk.Button(root)
        GButton2["bg"] = "#1653f9"
        ft = tkFont.Font(family='Times',size=10)
        GButton2["font"] = ft
        GButton2["fg"] = "#ffffff"
        GButton2["justify"] = "center"
        GButton2["text"] = "Start"
        GButton2.place(x=390,y=150,width=70,height=25)
        GButton2["command"] = root.destroy

        GButton3=tk.Button(root)
        GButton3["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton3["font"] = ft
        GButton3["fg"] = "#000000"
        GButton3["justify"] = "center"
        GButton3["text"] = "JGH"
        GButton3.place(x=260,y=230,width=70,height=25)
        GButton3["command"] = self.GButton3_command

        GButton4=tk.Button(root)
        GButton4["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton4["font"] = ft
        GButton4["fg"] = "#000000"
        GButton4["justify"] = "center"
        GButton4["text"] = "VAA"
        GButton4.place(x=450,y=230,width=70,height=25)
        GButton4["command"] = self.GButton4_command

        GButton5=tk.Button(root)
        GButton5["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=10)
        GButton5["font"] = ft
        GButton5["fg"] = "#000000"
        GButton5["justify"] = "center"
        GButton5["text"] = "HPR"
        GButton5.place(x=70,y=230,width=70,height=25)
        GButton5["command"] = self.GButton5_command

    def GButton3_command(self):
        with open('C:\\Users\\COMPUTER\\Desktop\\phyproject\\Attendance.csv', 'a+', newline='') as f:
            csvwriter = csv.writer(f,delimiter=" ")
            csvwriter.writerow("JGH")
            csvwriter.writerow("Date,Time,Name,Present")
        print("JGH")

    def GButton4_command(self):
        with open('C:\\Users\\COMPUTER\\Desktop\\phyproject\\Attendance.csv', 'a+', newline='') as f:
            csvwriter = csv.writer(f,delimiter=" ")
            csvwriter.writerow("VAA")
            csvwriter.writerow("Date,Time,Name,Present")
        print("VAA")

    def GButton5_command(self):
        with open('C:\\Users\\COMPUTER\\Desktop\\phyproject\\Attendance.csv', 'a+', newline='') as f:
            csvwriter = csv.writer(f,delimiter=" ")
            csvwriter.writerow("HPR")
            csvwriter.writerow("Date,Time,Name,Present")
        print("HPR")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

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
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

nameList=[]
def markAttendance(name):
    filename = "C:\\Users\\COMPUTER\\Desktop\\phyproject\\Attendance.csv"
    header = ["Date","Time", "Name", "Attendance"]
    found = False

    # get the current date
    now = datetime.now()
    today = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")
    # open the file in read mode to check if the student has already taken the attendance
    with open(filename, 'r+') as f:
        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            if row['Name'] == name and row['Date'] == today:
                found = True
                break
    if found:
        print(f"Error: Student with name '{name}' has already taken attendance today")
    else:
    # open the file in append mode to add data to the existing file
        with open(filename, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            # write the header only if the file is empty
            if os.stat(filename).st_size == 0:
                writer.writeheader()
            writer.writerow({"Date": today, "Time": time, "Name": name, "Attendance": 1})
            for student_name in classNames:
                if student_name != name:
                    writer.writerow({"Date": today, "Time": time, "Name": student_name, "Attendance": 0})
    
    print(name)

def check_exit():
    ok_to_exit = app.confirm_yesno('Confirm', 'Are you sure to continue to mark the Attendance', 'question')
    return ok_to_exit

encodeListKnown = findEncodings(images)
print('Encoding Complete')

## FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
cap = cv2.VideoCapture(0)
# while True:
# for i in range(10000):
for i in range(len(classNames)):
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0))
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # app = gp.GooeyPieApp('FRASC')
            # app.width = 250 
            # app.height =100
            # btn = gp.Button(app, 'Mark Attendance', markAttendance(name))
            # app.set_grid(2, 1)
            # app.add(btn, 1, 1, align='center')
            # app.on_close(check_exit)
            # app.run()
            # cv2.imshow('Webcam', img)
            # cv2.waitKey(10)
            markAttendance(name)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


