# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy, CountVectorizer and SVC from sklearn.
2. Create the email dataset and assign labels (Spam = 1, Not Spam = 0).
3. Convert email text into numerical features using CountVectorizer() and train the SVM model using fit().
4. 1. Predict whether the test email is spam or not using predict() and display the result.

## Program:
```py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

emails = [
    "Win a free prize now",
    "Claim your free gift",
    "Meeting scheduled for tomorrow",
    "Please attend the project meeting",
    "Congratulations you won a lottery",
    "Let's discuss the assignment"
]

labels = [1, 1, 0, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = SVC(kernel='linear')
model.fit(X, labels)

test_email = ["Win a free gift"]
test_vector = vectorizer.transform(test_email)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Spam Mail")
else:
    print("Not Spam Mail")

```

## Output:
<img width="130" height="46" alt="image" src="https://github.com/user-attachments/assets/d4d370c8-3140-4709-92fe-ce7011e59534" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
