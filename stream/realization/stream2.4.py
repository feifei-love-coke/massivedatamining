import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# numpy và pandas: Hỗ trợ xử lý dữ liệu.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('C:\\Users\\20505\\Downloads\\archive (1)\\archive1\\emails.csv')
print(data)
print(data.columns)
print(data.info())
print(data.isna().sum())
# 原代码此处因无Category列报错，现文件已有spam列，无需转换
# data['Spam'] = data['Category'].apply(lambda x: 1 if x =='spam' else 0)
print(data.head(5))

from sklearn.model_selection import train_test_split
# 修改为实际列名text和spam
X_train, X_test, y_train, y_test = train_test_split(data.text, data.spam, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

emails = [
    'How are you',
    'Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]
result = clf.predict(emails)
print(result)
score = clf.score(X_test, y_test)
print(score)