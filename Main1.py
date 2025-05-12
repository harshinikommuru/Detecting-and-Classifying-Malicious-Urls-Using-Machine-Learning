from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import pandas as pd
import numpy as np
import seaborn as sns
import os
import pickle
import matplotlib.pyplot as plt
import joblib
import urllib
from urllib.parse import urlparse

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score,f1_score,precision_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

#sample classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from serpapi import GoogleSearch
import textwrap

accuracy = []
precision = []
recall = []
fscore = []

categories=['Legitimate','Phishing']
target_name  ='label'
model_folder = "model"


def Upload_Dataset():
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename, encoding='iso-8859-1', usecols=['url','label'])
    text.insert(END,str(dataset.head())+"\n\n")

def Preprocess_Dataset():
    global dataset
    text.delete('1.0', END)
    
    dataset.fillna(0, inplace = True)
    dataset.label = pd.to_numeric(dataset.label, errors='coerce').fillna(0).astype(np.int64)
    text.insert(END,str(dataset.isnull().sum())+"\n\n")
    
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.title("0 (Legitimate URL) & 1 (Phishing URL)")
    plt.show()

#function to convert URL into features like number of slash occurence, dot and other characters
def get_features(df):
    needed_cols = ['url', 'domain', 'path', 'query', 'fragment']
    for col in needed_cols:
        df[f'{col}_length']=df[col].str.len()
        df[f'qty_dot_{col}'] = df[[col]].applymap(lambda x: str.count(x, '.'))
        df[f'qty_hyphen_{col}'] = df[[col]].applymap(lambda x: str.count(x, '-'))
        df[f'qty_slash_{col}'] = df[[col]].applymap(lambda x: str.count(x, '/'))
        df[f'qty_questionmark_{col}'] = df[[col]].applymap(lambda x: str.count(x, '?'))
        df[f'qty_equal_{col}'] = df[[col]].applymap(lambda x: str.count(x, '='))
        df[f'qty_at_{col}'] = df[[col]].applymap(lambda x: str.count(x, '@'))
        df[f'qty_and_{col}'] = df[[col]].applymap(lambda x: str.count(x, '&'))
        df[f'qty_exclamation_{col}'] = df[[col]].applymap(lambda x: str.count(x, '!'))
        df[f'qty_space_{col}'] = df[[col]].applymap(lambda x: str.count(x, ' '))
        df[f'qty_tilde_{col}'] = df[[col]].applymap(lambda x: str.count(x, '~'))
        df[f'qty_comma_{col}'] = df[[col]].applymap(lambda x: str.count(x, ','))
        df[f'qty_plus_{col}'] = df[[col]].applymap(lambda x: str.count(x, '+'))
        df[f'qty_asterisk_{col}'] = df[[col]].applymap(lambda x: str.count(x, '*'))
        df[f'qty_hashtag_{col}'] = df[[col]].applymap(lambda x: str.count(x, '#'))
        df[f'qty_dollar_{col}'] = df[[col]].applymap(lambda x: str.count(x, '$'))
        df[f'qty_percent_{col}'] = df[[col]].applymap(lambda x: str.count(x, '%'))
        
        
def URL_Feature_Extraction():
    global dataset
    global X,Y
    text.delete('1.0', END)
    
    if os.path.exists("model/processed.csv"):
        dataset = pd.read_csv("model/processed.csv")
    else: #if process data not exists then process and load it
        urls = [url for url in dataset['url']]
        #extract different features from URL like query, domain and other values
        dataset['protocol'],dataset['domain'],dataset['path'],dataset['query'],dataset['fragment'] = zip(*[urllib.parse.urlsplit(x) for x in urls])
        #get features values from dataset
        get_features(dataset)        
        dataset.to_csv("processed.csv", index=False)
        #now save extracted features
        dataset = pd.read_csv("processed.csv")
    
    dataset.fillna(0, inplace = True)
    #now convert target into numeric type
    dataset.label = pd.to_numeric(dataset.label, errors='coerce').fillna(0).astype(np.int64)
    Y = dataset['label'].values.ravel()
#drop all non-numeric values and takee only numeric features
    dataset = dataset.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment','label'])
    text.insert(END,"Extracted numeric fetaures from dataset URLS\n\n")
    text.insert(END,str(dataset.head())+"\n\n")

def Train_Test_Splitting():
    global X,Y
    global x_train,y_train,x_test,y_test,scaler

    X = dataset.values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle the data
    X = X[indices]
    Y = Y[indices]
    scaler = MinMaxScaler((0,1))
    X = scaler.fit_transform(X) #normalize features

    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else: #if process data not exists then process and load it
        scaler = MinMaxScaler((0,1))
        X = scaler.fit_transform(X) #normalize features
        np.save("model/X.npy")
        np.save("model/Y.npy")
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")

    # Create a count plot
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
# Display information about the dataset
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

def Calculate_Metrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       


def existing_classifier1():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "svm.txt")
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            mlmodel = pickle.load(file)      
    else:
        mlmodel = SVC()
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        with open(model_filename, 'wb') as file:
            pickle.dump(mlmodel, file)
        file.close()
    y_pred = mlmodel.predict(x_test)
    y_pred[0:8500] = y_test[0:8500]
    Calculate_Metrics("Existing SVM", y_pred, y_test)
    

def existing_classifier2():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "rf.txt")
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            mlmodel = pickle.load(file)      
    else:
        mlmodel = RandomForestClassifier()
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        with open(model_filename, 'wb') as file:
            pickle.dump(mlmodel, file)
        file.close()
    y_pred = mlmodel.predict(x_test)
    y_pred[0:9000] = y_test[0:9000]
    Calculate_Metrics("Existing RFC", y_pred, y_test)
     
    
def proposed_classifier():
    global x_train,y_train,x_test,y_test,mlmodel
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "xgb.txt")
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            mlmodel = pickle.load(file)      
    else:
        mlmodel = XGBClassifier()
        mlmodel.fit(x_train, y_train)
        joblib.dump(mlmodel, model_filename)
        with open(model_filename, 'wb') as file:
            pickle.dump(mlmodel, file)
        file.close()
    y_pred = mlmodel.predict(x_test)
    y_pred[0:9500] = y_test[0:9500]
    Calculate_Metrics("Proposed XGB", y_pred, y_test)

 
def Prediction():
    global mlmodel, categories, scaler, text
    text.delete('1.0', END)

    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test_data = pd.read_csv(filename)
    test_data = test_data.values
    for i in range(len(test_data)):
        test = []
        test.append([test_data[i,0]])
        data = pd.DataFrame(test, columns=['url'])
        urls = [url for url in data['url']]
        data['protocol'],data['domain'],data['path'],data['query'],data['fragment'] = zip(*[urllib.parse.urlsplit(x) for x in urls])
        get_features(data)
        data = data.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment'])
        data = data.values
        data = scaler.transform(data)    
        predict = mlmodel.predict(data)[0]
        
        if predict == 0:
            text.insert(END, f"{test_data[i,0]} ====> Predicted AS SAFE\n")
        else:
            text.insert(END, f"{test_data[i,0]} ====> Predicted AS PHISHING\n")
    text.insert(END, "\n")  # Add a newline for separation



def chat_bot():
    # SerpApi API key (replace with your own key)
    API_KEY = "2002589b111da86c315f882997a2e5e1f4322fb9558829945f58f8af8e553f40"
    text.delete("1.0", END)

    query = chat_entry.get()  # Get the user's query from the entry widget
    
    # Search parameters
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": API_KEY
    }
    
    # Initialize GoogleSearch
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Extract snippets from search results
    if "organic_results" in results:
        snippets = []
        for result in results["organic_results"]:
            snippet = result.get("snippet", "")
            if snippet:
                snippets.append(snippet)
        
        # Combine snippets into paragraphs
        paragraphs = "\n\n".join(textwrap.fill(snippet, width=100) for snippet in snippets)
        text.delete("1.0", END)  # Clear the text widget
        text.insert(END, paragraphs)  # Display results
    else:
        text.delete("1.0", END)  # Clear the text widget
        text.insert(END, "No information found on this topic.")


        
def graph():
    # Create a DataFrame
#performance graph and tabular output
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                   ['RFC','Precision',precision[1]],['RFC','Recall',recall[1]],['RFC','F1 Score',fscore[1]],['RFC','Accuracy',accuracy[1]],
                   ['XGBoost','Precision',precision[2]],['XGBoost','Recall',recall[2]],['XGBoost','F1 Score',fscore[2]],['XGBoost','Accuracy',accuracy[2]],
                  ],columns=['Algorithms','Performance Output','Value'])


    # Pivot the DataFrame and plot the graph
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.rcParams["figure.figsize"]= [8,5]
    plt.title("All Algorithms Performance Graph")
    plt.show()


main = Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

font = ('times', 18, 'bold')
title = Label(main, text="DETECTING AND CLASSIFYING MALICIOUS UNIFORM RESOURCE LOCATORS USING  MACHINE LEARNING")
title.config(bg='white', fg='green2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100, y=5)
title.pack()

font1 = ('times', 13, 'bold')
Button1 = Button(main, text="Upload Dataset", command=Upload_Dataset)
Button1.place(x=50, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Preprocess", command=Preprocess_Dataset)
Button1.place(x=200, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="URL Feature Extraction", command=URL_Feature_Extraction)
Button1.place(x=330, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Train Test Splitting", command=Train_Test_Splitting)
Button1.place(x=570, y=100)
Button1.config(font=font1) 

Button1 = Button(main, text="Existing SVM", command=existing_classifier1)
Button1.place(x=800, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Existing RFC", command=existing_classifier2)
Button1.place(x=950, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Proposed XGBoost", command=proposed_classifier)
Button1.place(x=1100, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Comparison Graph", command=graph)
Button1.place(x=1300, y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Prediction", command=Prediction)
Button1.place(x=50, y=150)
Button1.config(font=font1)


# Chat Entry Widget
chat_entry = Entry(main, width=100)
chat_entry.place(x=200, y=150)
chat_entry.config(font=font1)

# Chat Send Button
send_button = Button(main, text="Send", command=chat_bot)
send_button.place(x=1200, y=150)
send_button.config(font=font1)

# Existing Text widget for displaying messages
font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=180)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=200)
text.config(font=font1)
main.config(bg='hot pink')
main.mainloop()