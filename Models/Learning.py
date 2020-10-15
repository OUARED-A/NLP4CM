# Python3 code for preprocessing text 
import nltk 
import re 
import numpy as np 
import heapq 
import  pandas as pd
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, plot_confusion_matrix,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

# execute the text here as : 
data1 = pd.read_csv("dataTest.txt")
resultat = data1["1"]
#TARGET : les donnée sortée
y = resultat.iloc[:].values
print(y)
print(np.shape(y))

data1.columns = ["entrée","resultat"]
#LES DONN2E entrée
dataset =data1["entrée"]

#print(data["entrée"])

#print(data["entrée"])
for i in range(len(dataset)): 
    dataset[i] = dataset[i].lower() 
    patern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]))+')
    dataset[i] = patern.sub('', dataset[i])
    dataset[i] = re.sub(r"[,...\"!@#$%^&*(){}?;`~:<>+¨-]", " ", dataset[i])
    tokens = word_tokenize(dataset[i])
    table = str.maketrans(' ', ' ', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words=set(stopwords.words("french"))
    stop_words.discard("not")
    PS = PorterStemmer()
    words = [PS.stem(w) for w in words if not w in stop_words]
    words = ' '.join(words)
    dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])

# Creating the Bag of Words model 
word2count = {} 
for data in dataset: 
	words = nltk.word_tokenize(data) 
	for word in words: 
		if word not in word2count.keys(): 
			word2count[word] = 1
		else: 
			word2count[word] += 1

freq_words = heapq.nlargest(100, word2count, key=word2count.get)

vectorizer = CountVectorizer(ngram_range = (3,3))
X = vectorizer.fit_transform(dataset).toarray()
features = (vectorizer.get_feature_names()) 
print(features)
print(X)



x_train , x_test , y_train , y_test = train_test_split(X,y , test_size = 0.3 , random_state = 40)

print('4_SVC')
classifier4 = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.001 , C = 1000)
classifier4.fit(x_train, y_train)
y_pred4 = classifier4.predict(x_test)
print('accuracy :', accuracy_score(y_test,y_pred4))
print('f1 :', f1_score(y_test,y_pred4))
print('precision :', precision_score(y_test,y_pred4))
print(classification_report(y_test,y_pred4))
plot_confusion_matrix(classifier4, x_test, y_test) 


   
