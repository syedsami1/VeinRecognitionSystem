import tkinter
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageTk
from tkinter import filedialog
from sklearn import svm
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout, Activation, Flatten

global filename, canvas, text, images, root
global X, Y, filename, X_train, X_test, y_train, y_test, cnn_model
labels = ['vein1', 'vein2', 'vein3', 'vein4', 'vein5', 'vein6', 'vein7', 'vein8', 'vein9', 'vein10', 'vein11',
          'vein12', 'vein13', 'vein14', 'vein15', 'vein16', 'vein17', 'vein18', 'vein19', 'vein20',
          'vein21', 'vein22', 'vein23', 'vein24', 'vein25', 'vein26', 'vein27', 'vein28', 'vein29', 'vein30']

global accuracy, precision, recall, fscore

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    Y = np.load('model/Y.txt.npy')
    label = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21',
             'P22','P23','P24','P25','P26','P27','P28','P29','P30']
    unique, count = np.unique(Y, return_counts = True)
    height = count
    bars = label
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Person ID")
    plt.ylabel("Images Count")
    plt.title("Different Person Finger Vein found in Dataset")
    plt.show()

def preprocessDataset():
    global X, Y, filename, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32, 32, 3)
                    X.append(im2arr)
                    label = getID(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Total Images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    cv2.imshow("Sample Processed Image",cv2.resize(test,(150,250)))
    cv2.waitKey(0)

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    labels = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21',
             'P22','P23','P24','P25','P26','P27','P28','P29','P30']
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def runSVM():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
    y_train1 = np.argmax(y_train, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    X_train1 = X_train1[0:300,0:100]
    X_test1 = X_test1[0:100,0:100]
    y_train1 = y_train1[0:300]
    y_test1 = y_test1[0:100]
    svm_cls = svm.SVC()
    svm_cls.fit(X_train1, y_train1)
    predict = svm_cls.predict(X_test1)
    calculateMetrics("SVM Algorithm", y_test1, predict)

def runCNN():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore, cnn_model
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (2, 2), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (3, 3)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 16, epochs = 80, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        cnn_model = load_model("model/model_weights.hdf5")
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("CNN Algorithm", y_test1, predict)     

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['CNN','Precision',precision[1]],['CNN','Recall',recall[1]],['CNN','F1 Score',fscore[1]],['CNN','Accuracy',accuracy[1]],                                          
                  ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predictDisease(filename):
    global cnn_model
    text.delete('1.0', END)
    image = cv2.imread(filename)
    img = cv2.resize(image, (80, 80))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,80,80,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = cnn_model.predict(img)
    predict = np.argmax(preds)
    max_value = np.amax(preds)
    print(max_value)
    if max_value > 0.60:
        img = cv2.imread(filename)
        img = cv2.resize(img, (700,300))
        cv2.putText(img, 'Person Vein Identified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
        text.insert(END,'Person Vein Identified as : '+labels[predict]+"\n")        
    else:
        img = cv2.imread(filename)
        img = cv2.resize(img, (700,400))
        cv2.putText(img, 'Unable to identify person vein', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
        text.insert(END,'Unable to identify person vein : '+labels[predict])
    cv2.imwrite("images/output.png", img)        
    

def predict():
    global canvas, images, root
    filename = filedialog.askopenfilename(initialdir="testImages")
    predictDisease(filename)
    img = Image.open("images/output.png")
    img = img.resize((700, 300))
    picture = ImageTk.PhotoImage(img)
    canvas.configure(image = picture)
    canvas.image = picture
    root.update_idletasks()
    

def Main():
    global text, canvas, images, root

    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("Person Vein Identification using CNN")
    root.resizable(True,True)
    font = ('times', 14, 'bold')
    title = Label(root, text='Person Vein Identification using CNN')
    title.config(bg='yellow3', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)
    
    font1 = ('times', 12, 'bold')

    img = Image.open("images/download.png")
    img.resize((600, 300))
    picture = ImageTk.PhotoImage(img)
    canvas = Label(root, image=picture)
    canvas.place(x=300,y=200)

    uploadButton = Button(root, text="Upload Finger Vein Dataset", command=uploadDataset)
    uploadButton.place(x=60,y=80)
    uploadButton.config(font=font1)

    preprocessButton = Button(root, text="Preprocess Dataset", command=preprocessDataset)
    preprocessButton.place(x=400,y=80)
    preprocessButton.config(font=font1)

    svmButton = Button(root, text="Run SVM Algorithm", command=runSVM)
    svmButton.place(x=600,y=80)
    svmButton.config(font=font1)

    cnnButton = Button(root, text="Run CNN Algorithm", command=runCNN)
    cnnButton.place(x=60,y=130)
    cnnButton.config(font=font1)

    graphButton = Button(root, text="Comparison Graph", command=graph)
    graphButton.place(x=400,y=130)
    graphButton.config(font=font1)

    predictButton = Button(root, text="Identify Finger Vein from Test Image", command=predict)
    predictButton.place(x=600,y=130)
    predictButton.config(font=font1)

    text=Text(root,height=10,width=140)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=510)    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
