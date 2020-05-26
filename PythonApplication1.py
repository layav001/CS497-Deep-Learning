import numpy as np
import pandas as p
from tkinter import *
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sb
import joblib 

loaded_model = joblib.load('breastCancer.pkl')

#needed for data visualization
col_names = ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']
#this data is for reading breast cancer
data = p.read_csv("Breast_cancer_data.csv",header = None,names = col_names,skiprows = 6)


class MyFirstGUI:
    def __init__(self, master):
        self.clis = []
        self.master = master
        master.title("Breast Cancer Diagnostics")

        self.label = Label(master, text= "Radius: ",background = "black",foreground="Pink").grid(row = 0,column = 0,sticky = W)
        self.label = Label(master, text= "Texture: ",background = "black",foreground="Pink" ).grid(row = 1, column = 0,sticky = W)
        self.label = Label(master, text= "Perimeter: ",background = "black",foreground="Pink").grid(row = 2,column = 0,sticky = W)
        self.label = Label(master, text= "Area: " ,background = "black",foreground="Pink").grid(row = 3, column = 0,sticky = W)
        self.label = Label(master, text= "Smoothness: " ,background = "black",foreground="Pink").grid(row = 4, column = 0,sticky = W)

        self.texteditor = Text(master,width=60,height =10)
        self.texteditor.grid(row = 6,column = 10,sticky = W)
        self.vsb = Scrollbar(master, orient="vertical", command=self.texteditor.yview)#helps scroll through multiple outputs

        self.label = Label(master,text= "Name: ",background = "black",foreground="Pink").grid(row = 5,column=0,sticky=W)

        #This will get the user entry for radius
        self.Radiusentry = Entry(master)
        self.Radiusentry.grid(row=0, column=1, sticky =E) # this call returns NULL when it is done
        
        #will get the user entry for texture
        self.Textureentry = Entry(master)
        self.Textureentry.grid(row = 1,column = 1,sticky = E)

        #will get the user entry for perimeter
        self.Perimeterentry = Entry(master)
        self.Perimeterentry.grid(row=2, column=1, sticky =E) 

        #will get the user entry for Area
        self.Areaentry = Entry(master)
        self.Areaentry.grid(row = 3,column = 1,sticky = E)
        
        #will get the user entry for smoothness
        self.Smoothentry = Entry(master)
        self.Smoothentry.grid(row = 4,column = 1,sticky = E)

        #will get the user name
        self.name = Entry(master)
        self.name.grid(row = 5,column = 1,sticky = E)

        #list of buttons used for a variety of things
        self.results = Button(master,text = "Results for Dignosis",command=self.results,fg="pink",bg="black").grid(row=4,column=10,sticky=W)
        self.printGraph = Button(master,text = "Show heat Map",command=self.plot_data,bg="red",fg="black").grid(row=1,column=10,sticky = W)
        self.graph = Button(master,text = "Bar graph",command = self.graph_results).grid(row=2,column=10,sticky=W)
        self.pgraph = Button(master,text = "Pair Plot",command = self.pairplot).grid(row=3,column=10,sticky=W)
        self.close = Button(master,text = "Quit",command = self.quit).grid(row=5,column=10,sticky=W)

    def quit(self):
        self.master.destroy()

    def plot_data(self):
        sb.set(style='ticks',color_codes=True)
        plt.figure(figsize=(12,8))
        sb.heatmap(data.astype(float).corr(),linewidths=0.1,square=True,linecolor='white',annot=True)
        plt.show()

    #takes user entry and dislays what the diagnsotics was
    def results(self):
        if self.Radiusentry.get() and self.Textureentry.get() and self.Perimeterentry.get() and self.Areaentry.get() and self.Smoothentry.get():
            num = float(self.Radiusentry.get())
            num1 = float(self.Textureentry.get())
            num2 = float(self.Perimeterentry.get())
            num3 = float(self.Areaentry.get())
            num4 = float(self.Smoothentry.get())
            person = {'mean_radius': num,'mean_texture': num1,'mean_perimeter': num2,'mean_area': num3,'mean_smoothness' : num4}
            index = [1]
            person_info = p.DataFrame(person,index)
            person_info
            y_pred = loaded_model.predict(person_info)
            if(y_pred == 0):
                person = "The Diagnosis for " + self.name.get() + " is NO Cancer was Detected!" + "\n"
                self.texteditor.insert(END,person)
            else:
                person2 = "The Diagnosis for " + self.name.get() + " is Cancer was Detected see your Doctor as soon as possible!" + "\n"
                self.texteditor.insert(END,person2)
        else:
            self.texteditor.insert(END,"Error not enough results entered or wrong information!" + "\n")

    #display graphs with number of people that were diagnosed with cancer 
    def graph_results(self):
        data['diagnosis'].value_counts()
        sb.countplot(data['diagnosis'],label="Count")

    #display graphs that compare the other features with diagnostics
    def pairplot(self):
        sb.pairplot(data,hue="diagnosis")

root = Tk()
root['bg'] = 'pink'
my_gui = MyFirstGUI(root)
root.mainloop()  

