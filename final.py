import os
from tkinter import *
from PIL import ImageTk,Image
from imageai.Detection import ObjectDetection
import os
import cv2
import matplotlib.pyplot as plt

execution_path=os.getcwd()


#object Detection part

detector=ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()
custom=detector.CustomObjects(person=True,motorcycle=True)
detection=detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=os.path.join(execution_path,"im.jpg")
                                                ,output_image_path=os.path.join(execution_path,"out.jpg"),minimum_percentage_probability=50)

def object_detection():
    img=cv2.imread('out.jpg')
    inn=cv2.imread("im.jpg")
    person=[]
    for eachObject in detection:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        p=eachObject["box_points"]
        #plt.imshow(inn[p[1]:p[0],p[3]:p[2]])
        point=eachObject["name"]
        if point=="person":
            person.append(inn[p[1]:p[0],p[3]:p[2]])
        #plt.show()
        for i in range(len(person)):
            plt.imsave("C:\\Users\Sahil\Desktop\Minor\object detection\object detection\image\save\image"+str(i)+".jpg",person[i])
  
#end of object detection

def vide():
    os.startfile("C:\\Users\Sahil\Desktop\Minor\object detection\object detection\image\save")

#GUI part

root=Tk("Project")
bu1=PhotoImage(file="C:/Users/Sahil/Desktop/data/bu.png")
bu2=PhotoImage(file="C:/Users/Sahil/Desktop/data/open.png")

#set width and height 
lb=Label(root,text="Helmet Detection",anchor=CENTER,font=('Times Roman',30))
lb.pack()


#background image code
canvas=Canvas(root,width=800,height=500)
image=ImageTk.PhotoImage(Image.open("C:\\Users\\Sahil\\Desktop\\data\\lo.png"))
canvas.create_image(0,0,anchor=NW,image=image)
canvas.pack()

#button 1
button1 = Button(root, image=bu1, command=object_detection,border=0)
button1.pack(side="left", fill='both', expand=True, padx=4, pady=4)


#button 2 code
button = Button(root, image=bu2, command=vide,border=0)
button.pack(side="right", fill='both', expand=True, padx=4, pady=4)





root.mainloop()

#end of gui
