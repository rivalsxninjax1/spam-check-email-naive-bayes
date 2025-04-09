import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# machnine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#loading the data set
df = pd.read_csv('src/data.csv', encoding='latin-1', sep='\t', header=None, names=['Label', 'Message'])



#displaying the first five rows
print("first five rows of the data set ")
print(df.head())

print("\n Dataset shape", df.shape)
print("columns:", df.columns.to_list())


plt.figure(figsize=(8, 4))
sns.countplot(x='Label', data=df, palette="Set2")
plt.title("Distribution of sms labels (spam vs ham)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

#creating columns for length of each message 
df['Message_Length']= df['Message'].apply(len)

#plot histogram for message lenght 
plt.figure(figsize=(8, 4))
sns.histplot(df['Message_Length'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of message lenghts")
plt.xlabel("message lenght")
plt.ylabel("frequency")
plt.show()



#text processing and feature extraction
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['Message'])
y = df['Label']

print("sample of the feature names:", vectorizer.get_feature_names_out()[:10])

#reint test splits and model training 
#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("training set shape:", X_train.shape)
print("test shape shape:",X_test.shape)

#model training 
model = MultinomialNB()

#training the classifier on the traiing data 
model.fit(X_train, y_train)

print("model traiing complete")

#model evaluation 
y_pred = model.predict(X_test)

#evaluate the model perfromance
accuracy = accuracy_score(y_test, y_pred)
print("accuracy on the test set: {:.2f}".format(accuracy * 100))

#detailed classifictaion report 
print("\n classification report")
print(classification_report(y_test, y_pred))

#computing the confusion matrix

cm = confusion_matrix(y_test, y_pred)
print("\n confusion matrix")
print(cm)

#visualizing the confusion matrix

plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.show()

#predicting the new sms message
#define new sms messages for prdiction
new_messages = [
    "Congratulations! You have won a lottery ticket. Claim your prize by calling now.",
    "Hey, are we still meeting for dinner tonight?"
]
#transfer the new message using the exixting vectorizer 
X_new = vectorizer.transform(new_messages)

#use the clasifier to predict the labels for new messages 
new_predictions = model.predict(X_new)

#print out the predictions 

for msg, pred in zip(new_messages, new_predictions):
    print("message:", msg)
    print("Predicted Label:", pred)
    print("--------")





