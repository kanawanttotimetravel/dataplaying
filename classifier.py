import os.path

from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, RocCurveDisplay, auc, PrecisionRecallDisplay

# Download 10000 samples from cc100
if not os.path.isfile('data/cc100-samples.csv'):
    dataset = load_dataset("eson/cc100-samples", 'vi')
    cc_df = dataset['train'].to_pandas()
    cc_df.to_csv('data/cc100-samples.csv')

# High quality dataset
hq_df = pd.read_csv('data/test_data.csv')
hq_df = hq_df.dropna()
# Only choose knowledge-intensive subject to avoid latex
samples = hq_df[hq_df['subject'].isin(['ngu-van', 'gdcd', 'lich-su', 'dia-ly'])]
hq_list = samples['text'].to_list()
hq_list = [' '.join(text.split()) for text in hq_list]

# Low quality dataset
lq_df = pd.read_csv('data/cc100-samples.csv')
# Split into chunks then merge
chunks = np.array_split(lq_df['text'].to_list(),1000)
lq_list = [' '.join(chunk) for chunk in chunks]

# labeled data
data = {'text': [], 'label': []}
for text in hq_list:
    data['text'].append(text)
    data['label'].append(1)

for text in lq_list:
    data['text'].append(text)
    data['label'].append(0)

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
train, test = train_test_split(shuffle(df), test_size=0.2)


# Accuracy checking
def accuracy_check(classifier):
    X_train = vectorizer.fit_transform(train['text'])
    y_train = train['label']

    X_test = vectorizer.transform(test['text'])
    y_test = test['label']

    model = classifier
    model.fit(X_train, y_train)
    acc = accuracy_score(model.predict(X_test), y_test)
    print(f'Accuracy of {classifier} = {acc * 100}')

    # Get FP/TP rate
    scores = model.predict_proba(X_test)[:,1]
    # print(scores)
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=classifier, )
    roc.plot()

    # Plot precision-recall curve
    precision_recall = PrecisionRecallDisplay.from_predictions(y_test, scores, name=classifier)
    # precision_recall.plot()
    plt.show()


# accuracy_check(classifier=svm.SVC(probability=True))
accuracy_check(classifier=RandomForestClassifier())

# X = vectorizer.fit_transform(df['text'])
# y = df['label']
# SVMclassifier = svm.SVC()
# SVMclassifier.fit(X, y)

