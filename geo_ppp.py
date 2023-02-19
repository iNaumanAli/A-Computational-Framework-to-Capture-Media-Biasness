# -*- coding: utf-8 -*-
"""GEO PPP.ipynb



# **GEO PPP**

## 1. Installing Libraries
"""

pip install --upgrade matplotlib

pip install pandas nltk wordcloud

"""## 2.   Importing Libraries"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

from wordcloud import WordCloud, STOPWORDS
import re # Regular expression
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import warnings
warnings.filterwarnings("ignore")

"""## 3. Google Drive Authorization


"""

# Google Colab Setup
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive', force_remount=True)

"""## 4. Google Drive Path"""

path = "/content/drive/Othercomputers/My Laptop/Thesis Data/After Cleaning/AC GEO PPP.csv"
data = pd.read_csv(path,encoding= 'unicode_escape')
data

"""## 10. Maping Function of POS Tag"""

def get_wordnet_pos(word):
  """Map POS tag to first character lemmatize() accepts"""
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {"J": wordnet.ADJ,
              "N": wordnet.NOUN,
              "V": wordnet.VERB,
              "R": wordnet.ADV}
  return tag_dict.get(tag, wordnet.NOUN)

"""## 11. Data Cleaning"""

# Cleaning the tweets
def cleanTweets(tweet):
    
    # Convert all tweets to lowercase
    tweet = tweet.lower()
    
    # Remove URL
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    
    # Remove @user and '#' tags from tweet
    tweet = re.sub(r'\@\w+|\#\w+', "", tweet)
    
    # Remove punctuations
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    #filtered_words = [word for word in tweet_tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_words = ' '.join([lemmatizer.lemmatize (w, get_wordnet_pos(w)) for w in tweet_tokens])
    
    # Remove Non-English Characters
    encoded_string = lemma_words.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    
    return decode_string

#cleanTweets("Stay Strong Lahore. ðŸ’”  #LahoreBlast #PrayForLahore https://t.co/fNVWKbo3B3")

# Apply function
data['Tweets'] = data['Tweets'].apply(cleanTweets)

data

"""## Remove Empty Cells"""

data = data[data['Tweets'].notna()]
print (data.isnull().sum())

"""## Save Cleaned File"""

#Saving it to the csv file 
data.to_csv('/content/drive/Othercomputers/My Laptop/Thesis Data/After Cleaning/AC GEO PMLN.csv',index=False)

"""## 12. Text to Biasness"""

import nltk
nltk.download('vader_lexicon')
#importing sentimentintensityanaylzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
data['scores'] = data['Tweets'].apply(lambda Tweets: sid.polarity_scores(Tweets))
def sentimentPredict(sentiment):
  if sentiment['compound'] >= 0.05:
    return "Positive"
  elif sentiment['compound'] <= -0.05: 
    return "Negative"
  else:
    return "Neutral"
data['label'] =data['scores'].apply(lambda x: sentimentPredict(x))
data.head(10)

"""## Save Labeled File"""

#Saving it to the csv file 
data.to_csv('/content/drive/Othercomputers/My Laptop/Thesis Data/After Labeling/AL GEO PPP.csv',index=False)

"""## Read Labeled File"""

path = "/content/drive/Othercomputers/My Laptop/Thesis Data/After Labeling/AL GEO PPP.csv"
data = pd.read_csv(path)
data

"""## Emotions Count Graph"""

count_values = data['Analysis'].value_counts()
sns.set(style = "whitegrid", font_scale = 1.1)
plt.figure(figsize = (9,6))
color = sns.color_palette('bright')
ax = sns.barplot(count_values.index, count_values.values, palette = color)
ax.set(xlabel = 'Labels', ylabel = 'Labels Count')
for container in ax.containers:
  ax.bar_label(container)
plt.title("ARY PMLN Bar", fontsize = 16)
plt.savefig("/content/drive/Othercomputers/My Laptop/Thesis Data/Results Images/ARY PMLN Bar.png")

"""## Emotions Percentage Graph"""

data = data[data['Tweets'].notna()]
print (data.isnull().sum())

data2 = data["Analysis"].value_counts()
color = sns.color_palette('bright')
explode = [0.01, 0.01, 0.01]
data2.plot.pie(autopct = "%.2f%%", colors = color, fontsize = 14, explode = explode, figsize = (10,8))
plt.axis('off')
plt.title("ARY PMLN Percentage", fontsize = 16)
plt.legend()
#plt.axis('equal')
plt.savefig("/content/drive/Othercomputers/My Laptop/Thesis Data/Results Images/ARY PMLN Percent.png")

"""## Machine Learning Models"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline


def train_model(model, data, targets):
    text_clf = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb',ngram_range=(1,4))),
    ('tfidf', TfidfTransformer()),
    ('clf', model),
    ])
    text_clf.fit(data, targets)
    return text_clf
def get_accuracy(trained_model,X, y):
    predicted = trained_model.predict(X)
    accuracy = np.mean(predicted == y)
    return accuracy
def get_report(trained_model,X, y, model_name):
    plt.figure(figsize=(10,10))
    predicted = trained_model.predict(X)
    cr_report = classification_report(y,predicted,target_names=enc.classes_, output_dict=True)
    cr = classification_report(y,predicted,target_names=enc.classes_)
    cm = confusion_matrix(y,predicted)
    index = ['Positive', 'Negative', 'Neutral']
    columns = ['Positive', 'Negative', 'Neutral']
    df_cm = pd.DataFrame(cm,columns,index)
    #df_cm=pd.DataFrame(cm,index=enc.classes_)
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt='g',cmap="OrRd")
    plt.title('ARY PMLN '+model_name+" Confusion Matrix", fontsize = 16)
    plt.savefig('/content/drive/Othercomputers/My Laptop/Thesis Data/Results Images/ARY PMLN '+model_name+' Confusion Matrix.png')
    plt.show()
    global macro_precision
    global macro_recall
    global macro_f1
    macro_precision =  cr_report['macro avg']['precision']
    macro_recall = cr_report['macro avg']['recall']    
    macro_f1 = cr_report['macro avg']['f1-score']
    print(cr)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X=data['Tweets']
Y=data['Analysis']
data.shape

enc=LabelEncoder()
Y=enc.fit_transform(Y)

X_train , X_test ,y_train,y_test = train_test_split(X,Y,test_size=0.3)

"""## Logistic Regression"""

from sklearn.linear_model import LogisticRegression
trained_clf_LogisticRegression = train_model(LogisticRegression(), X_train, y_train)
accuracy = get_accuracy(trained_clf_LogisticRegression,X_test, y_test)*100
print(f"Test dataset accuracy with Logistic Regression: {accuracy:.2f}" + " %")
get_report(trained_clf_LogisticRegression,X_test, y_test,model_name = 'Logistic Regression')

LR_Accuracy = round(accuracy,0)
LR_Precision = round((macro_precision)*100)
LR_Recall = round((macro_recall)*100)
LR_F1_Score = round((macro_f1)*100)

"""## Naive Bayes"""

from sklearn.naive_bayes import MultinomialNB
trained_clf_MultinomialNB = train_model(MultinomialNB(), X_train, y_train)
accuracy = get_accuracy(trained_clf_MultinomialNB,X_test, y_test)*100
print(f"Test dataset accuracy with Naive Bayes: {accuracy:.2f}" + " %")
get_report(trained_clf_MultinomialNB,X_test, y_test,model_name = 'Naive Bayes')

NB_Accuracy = round(accuracy,0)
NB_Precision = round((macro_precision)*100)
NB_Recall = round((macro_recall)*100)
NB_F1_Score = round((macro_f1)*100)

"""## SVM"""

from sklearn.svm import SVC
trained_clf_linearSVC = train_model(SVC(), X_train, y_train)
accuracy = get_accuracy(trained_clf_linearSVC,X_test, y_test)*100
print(f"Test dataset accuracy with SVM: {accuracy:.2f}"  + " %")
get_report(trained_clf_linearSVC,X_test, y_test,model_name = 'SVM')

SVM_Accuracy = round(accuracy,0)
SVM_Precision = round((macro_precision)*100)
SVM_Recall = round((macro_recall)*100)
SVM_F1_Score = round((macro_f1)*100)

"""## Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier
trained_clf_DT = train_model(DecisionTreeClassifier(), X_train, y_train)
accuracy = get_accuracy(trained_clf_DT,X_test, y_test)*100
print(f"Test dataset accuracy with Decision Tree Classifier: {accuracy:.2f}" + " %")
get_report(trained_clf_DT,X_test, y_test,model_name = 'Decision Tree Classifier')

DTC_Accuracy = round(accuracy,0)
DTC_Precision = round((macro_precision)*100)
DTC_Recall = round((macro_recall)*100)
DTC_F1_Score = round((macro_f1)*100)

"""## Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
trained_clf_RF = train_model(RandomForestClassifier(n_estimators=200), X_train, y_train)
accuracy = get_accuracy(trained_clf_RF,X_test, y_test)*100
print(f"Test dataset accuracy with Random Forest Classifier: {accuracy:.2f}" + " %")
get_report(trained_clf_RF,X_test, y_test,model_name = 'Random Forest Classifier')

RFC_Accuracy = round(accuracy,0)
RFC_Precision = round((macro_precision)*100)
RFC_Recall = round((macro_recall)*100)
RFC_F1_Score = round((macro_f1)*100)

"""## K-Nearest Neighbors (KNN)"""

from sklearn.neighbors import KNeighborsClassifier
trained_clf_KNN = train_model(KNeighborsClassifier(n_neighbors=3), X_train, y_train)
accuracy = get_accuracy(trained_clf_KNN,X_test, y_test)*100
print(f"Test dataset accuracy with KNN: {accuracy:.2f}" + " %")
get_report(trained_clf_KNN,X_test, y_test,model_name = 'KNN')

KNN_Accuracy = round(accuracy,0)
KNN_Precision = round((macro_precision)*100)
KNN_Recall = round((macro_recall)*100)
KNN_F1_Score = round((macro_f1)*100)

"""## Models Performance Comparison"""

d = {'Model': ['Logistic Regression', 'Naive Bayes','SVM','Decision Tree', 'KNN'], 'Precision': [LR_Precision, NB_Precision, SVM_Precision, DTC_Precision, KNN_Precision], 'Recall':[LR_Recall, NB_Recall, SVM_Recall, DTC_Recall, KNN_Recall], 'F1-Score':[LR_F1_Score, NB_F1_Score, SVM_F1_Score, DTC_F1_Score, KNN_F1_Score], 'Accuracy':[LR_Accuracy, NB_Accuracy, SVM_Accuracy, DTC_Accuracy, KNN_Accuracy]} 
df = pd.DataFrame(data=d)
df = df.melt(id_vars=['Model'], value_vars=['Precision', 'Recall','F1-Score','Accuracy'], value_name='Percentage Score', var_name='')
sns.set(style = "whitegrid", font_scale = 1.1)
plt.figure(figsize = (14,8))
color = sns.color_palette('bright')
plt.title("GEO PPP Models Performance Comparison", fontsize = 16)
chart = sns.barplot(x='Model', y='Percentage Score', hue='', data=df, palette = color)
plt.legend(loc='center right')
for container in chart.containers:
  chart.bar_label(container)
plt.savefig("/content/drive/Othercomputers/My Laptop/Thesis Data/Results Images/GEO PPP Models Performance Comparison.png")
plt.show()

"""## Word Cloud"""

# generate the word cloud
wordcloud = WordCloud(background_color='white', width=1600, height=800, random_state = 42).generate(' '.join(data['Tweets']))
# plot the word cloud
plt.figure(figsize=(16,10), frameon=False)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("/content/drive/Othercomputers/My Laptop/Thesis Data/Results Images/GEO PPP Word Cloud.png")
plt.show()
