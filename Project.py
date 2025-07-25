import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w

w.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("AI-Data.csv")

# Menu for visualization
ch = 0
while ch != 10:
    print("\n1.Marks Class Count Graph\t2.Semester-wise\n3.Gender-wise\t\t\t4.Nationality-wise\n5.Grade-wise\t\t\t6.Section-wise\n7.Topic-wise\t\t\t8.Stage-wise\n9.Absent Days-wise\t\t10.No Graph\n")
    ch = int(input("Enter Choice: "))
    
    if ch == 1:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.title("Class Count")
        plt.show()
        
    elif ch == 2:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Semester")
        plt.show()
        
    elif ch == 3:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'])
        plt.title("Class vs Gender")
        plt.show()
        
    elif ch == 4:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Nationality")
        plt.xticks(rotation=90)
        plt.show()
        
    elif ch == 5:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='GradeID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Grade")
        plt.show()
        
    elif ch == 6:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Section")
        plt.show()
        
    elif ch == 7:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Topic")
        plt.xticks(rotation=90)
        plt.show()
        
    elif ch == 8:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Stage")
        plt.show()
        
    elif ch == 9:
        print("Loading...\n"); t.sleep(1)
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'])
        plt.title("Class vs Absence Days")
        plt.show()
        
if ch == 10:
    print("Exiting...\n"); t.sleep(1)

# Drop unwanted columns
columns_to_drop = [
    'gender', 'StageID', 'GradeID', 'NationalITy', 'PlaceofBirth',
    'SectionID', 'Topic', 'Semester', 'Relation',
    'ParentschoolSatisfaction', 'ParentAnsweringSurvey', 'AnnouncementsView'
]
data = data.drop(columns=columns_to_drop)

# Shuffle dataset
u.shuffle(data)

# Label encode categorical columns
for column in data.columns:
    if data[column].dtype == object:
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Split features and labels
X = data.drop("Class", axis=1).values
y = data["Class"].values

# Encode target labels
le_y = pp.LabelEncoder()
y = le_y.fit_transform(y)  # H=0, M=1, L=2

# Train/Test Split (70/30)
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train models
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "Neural Network (MLP)": nn.MLPClassifier(activation="logistic")
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy Measures - {name}")
    print(m.classification_report(y_test, y_pred, target_names=le_y.classes_))
    print(f"Accuracy: {round(accuracy, 3)}\n")
    t.sleep(1)
