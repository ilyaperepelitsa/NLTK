import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import codecs


from unidecode import unidecode

corpus = "".join(words)
sent = []
sent.append(unidecode("".join(corpus)))


def preprocessing(text):
    text = str(text)
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # remove stopwords
    stop = stopwords.words("english")
    tokens = [token for token in tokens if token not in stop]


    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text


sms = open("/Users/ilyaperepelitsa/Downloads/smsspamcollection/SMSSpamCollection", encoding = "utf-8" )
sms_data = []
sms_labels = []
csv_reader = csv.reader(sms, delimiter = "\t")


for line in csv_reader:
    #adding the sms_id
    sms_labels.append(line[0])

    #adding the cleaned text
    sms_data.append(preprocessing(line[1]))
sms.close()

import sklearn
import numpy as np

trainset_size = int(round(len(sms_data)*0.7))
print("The training set size for this classifier is " + str(trainset_size) + "\n")
x_train = np.array(["".join(el) for el in sms_data[0:trainset_size]])
y_train = np.array([el for el in sms_labels[0:trainset_size]])

x_test = np.array(["".join(el) for el in sms_data[trainset_size+1:len(sms_data)]])
y_test = np.array([el for el in sms_labels[trainset_size+1:len(sms_labels)]])


from sklearn.feature_extraction.text import CountVectorizer
sms_exp = []
for line in csv_reader:
    sms_exp.append(preprocessing(line[1]))

vectorizer = CountVectorizer(min_df = 1)
X_exp = vectorizer.fit_transform(sms_exp)
print("||".join(vectorizer.get_feature_names()))
print(X_exp.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 2, ngram_range = (1, 2), stop_words = "english",
                                strip_accents = "unicode", norm = "l2")

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
clf = MultinomialNB().fit(X_train, y_train)
y_nb_predicted = clf.predict(X_test)
print(y_nb_predicted)
cm = confusion_matrix(y_test, y_nb_predicted)
print(cm)


feature_names = vectorizer.get_feature_names()
coefs = clf.coef_
intercept = clf.intercept_

coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n = 10
top = zip(coefs_with_fns[:n], coefs_with_fns[:(-n + 1): -1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


#CART
from sklearn.metrics import classification_report
from sklearn import tree
clf = tree.DecisionTreeClassifier().fit(X_train.toarray(), y_train)
y_tree_predicted = clf.predict(X_test.toarray())
print(y_tree_predicted)
print(classification_report(y_test, y_tree_predicted))



#SGD EXAMPLE
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

clf = SGDClassifier(alpha = 0.0001, n_iter = 50).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_pred, y_test)
print(cm)


feature_names = vectorizer.get_feature_names()
coefs = clf.coef_
intercept = clf.intercept_

coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n = 20
top = zip(coefs_with_fns[:n], coefs_with_fns[:(-n + 1): -1])

for (coef_1, fn_1), (coef_2, fn_2) in top:
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))



from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
y_svm_predicted = svm_classifier.predict(X_test)
print(classification_report(y_test, y_svm_predicted))
cm = confusion_matrix(y_test, y_pred)
print(cm)



feature_names = vectorizer.get_feature_names()
coefs = svm_classifier.coef_
intercept = svm_classifier.intercept_

coefs_with_fns = sorted(zip(svm_classifier.coef_[0], feature_names))
n = 20
top = zip(coefs_with_fns[:n], coefs_with_fns[:(-n + 1): -1])

for (coef_1, fn_1), (coef_2, fn_2) in top:
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))




from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(n_estimators = 10).fit(X_train, y_train)
predicted = RF_clf.predict(X_test)
print(classification_report(y_test, predicted))
cm = confusion_matrix(y_test, predicted)
print(cm)





# BATCHES KMEANS

from sklearn.cluster import KMeans, MiniBatchKMeans
true_k = 5
km = KMeans(n_clusters = true_k, init = "k-means++", max_iter = 100, n_init = 1)
kmini = MiniBatchKMeans(n_clusters = true_k, init = "k-means++", n_init = 1,
                    init_size = 1000, batch_size = 1000, verbose = True)
km_model = km.fit(X_train)
kmini_model = kmini.fit(X_train)

# clustering = collections.defaultdict(list)
for idx, label in enumerate(kmini_model.labels_):
    print(str(label) + str(idx))
