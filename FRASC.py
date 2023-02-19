import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
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
path1 = 'phyproject'
def add_images():
    directory = filedialog.askdirectory()
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            shutil.copy(image_path, path)

def test_img():
    filename = filedialog.askopenfilename()
    try:
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # if os.path.exists("image.jpg"):
        #     os.remove("image.jpg")
            os.rename(filename, "image.jpg")
    except:
        print("Image uploaded")

class App:
    def __init__(self, root):
        #setting title
        root.title("FRASC")
        #setting window size
        root.geometry("1001x601")
        root.resizable(width=True, height=True)
        root.config(bg="#3EB489")

        GLabel=tk.Label(root)
        GLabel["activeforeground"] = "#1e9fff"
        GLabel["bg"] = "#1cd550"
        ft = tkFont.Font(family='Times',size=30)
        GLabel["font"] = ft
        GLabel["fg"] = "#c71585"
        GLabel["justify"] = "center"
        GLabel["text"] = "FRASC  : THE REVOLUTION"
        GLabel.place(x=10,y=20,width=979,height=63)
        
        GButton_1=tk.Button(root)
        GButton_1["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GButton_1["font"] = ft
        GButton_1["fg"] = "#000000"
        GButton_1["justify"] = "center"
        GButton_1["text"] = "JGH"
        GButton_1["relief"] = "flat"
        GButton_1.place(x=100,y=400,width=92,height=30)
        GButton_1["command"] = self.GButton1_command

        GButton_2=tk.Button(root)
        GButton_2["bg"] = "#f6f5f5"
        ft = tkFont.Font(family='Times',size=10)
        GButton_2["font"] = ft
        GButton_2["fg"] = "#000000"
        GButton_2["justify"] = "center"
        GButton_2["text"] = "HPR"
        GButton_2.place(x=440,y=400,width=92,height=30)
        GButton_2["command"] = self.GButton2_command

        GButton_3=tk.Button(root)
        GButton_3["bg"] = "#5fb878"
        ft = tkFont.Font(family='Times',size=10)
        GButton_3["font"] = ft
        GButton_3["fg"] = "#000000"
        GButton_3["justify"] = "center"
        GButton_3["text"] = "VAA"
        GButton_3.place(x=800,y=400,width=92,height=30)
        GButton_3["command"] = self.GButton3_command

        GButton_4=tk.Button(root)
        GButton_4["activebackground"] = "#ff5722"
        GButton_4["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GButton_4["font"] = ft
        GButton_4["fg"] = "#000000"
        GButton_4["justify"] = "center"
        GButton_4["text"] = "ADD IMAGES"
        GButton_4.place(x=260,y=200,width=92,height=30)
        GButton_4["command"] = add_images

        GButton_5=tk.Button(root, command=lambda: [test_img(), root.destroy()])
        GButton_5["bg"] = "#ff5722"
        ft = tkFont.Font(family='Times',size=10)
        GButton_5["font"] = ft
        GButton_5["fg"] = "#000000"
        GButton_5["justify"] = "center"
        GButton_5["text"] = "TAKE ATTENDANCE"
        GButton_5.place(x=640,y=200,width=130,height=37)
        
    def GButton1_command(self):
        with open('Attendance.csv', 'a+', newline='') as f:
            csvwriter = csv.writer(f,delimiter=" ")
            csvwriter.writerow("JGH")
            csvwriter.writerow("Date,Time,Name,Attendance")
        print("JGH")

    def GButton2_command(self):
        with open('Attendance.csv', 'a+', newline='') as f:
            csvwriter = csv.writer(f,delimiter=" ")
            csvwriter.writerow("HPR")
            csvwriter.writerow("Date,Time,Name,Attendance")
        print("HPR")

    def GButton3_command(self):
        with open('Attendance.csv', 'a+', newline='') as f:
            csvwriter = csv.writer(f,delimiter=" ")
            csvwriter.writerow("VAA")
            csvwriter.writerow("Date,Time,Name,Attendance")
        print("VAA")


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


def markAttendance(name,nameList):
    nameList.append(name)
    filename = "Attendance.csv"
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
            for name in nameList:
                writer.writerow({"Date": today, "Time": time, "Name": name, "Attendance": 1})
            # for stu in classNames:
            #     if stu in nameList:
            #         writer.writerow({"Date": today, "Time": time, "Name": name, "Attendance": 1})
            #     else:
            #         writer.writerow({"Date": today, "Time": time, "Name": stu, "Attendance": 0})
                
    print(name)

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# cap = cv2.VideoCapture(0)

img = cv2.imread("image.jpg")
nameList=[]
# while True:
# for i in range(10000):
for i in range(len(classNames)):
    # success, img = cap.read()
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
            # y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0))
            # cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # # app = gp.GooeyPieApp('FRASC')
            # app.width = 250 
            # app.height =100
            # btn = gp.Button(app, 'Mark Attendance', markAttendance(name))
            # app.set_grid(2, 1)
            # app.add(btn, 1, 1, align='center')
            # app.run()
            # cv2.imshow('Webcam', img)
            # cv2.waitKey(10)
            namelist=list(set(nameList))
            markAttendance(name,namelist)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     cap.release()
        #     break






