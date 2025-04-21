import pandas as pd
import spacy
import string
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_csv('completeSpamAssassin.csv')
data = data.dropna()
data = data.sample(frac=1).reset_index(drop=True)
data = data[0:1000]

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation

def spacy_tokenizer(sentence):
    if isinstance(sentence, float):
        return ""
    doc = nlp(str(sentence))
    mytokens = [word.lemma_.lower().strip() for word in doc]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return " ".join(mytokens)

data['tokenized_Body'] = data['Body'].apply(spacy_tokenizer)
x = data['tokenized_Body']
y = data['Label']

vec = CountVectorizer()
X_vec = vec.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y)

model = XGBClassifier(n_estimators=250)
model.fit(x_train, y_train)

y_pred_proba = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

class_names = ['ham', 'spam']
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('XGBoost ROC Curve')
plt.legend(loc="lower right")
plt.show()