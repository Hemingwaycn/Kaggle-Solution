import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 舍去木有信息
drop_features = ['PassengerId', 'Survived']
train_drop = train.drop(drop_features, axis=1)

titanic = pd.concat([train, test], sort=False)
len_train = train.shape[0]

# TODO 特征工程
# 提取尊称
titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

# 把少数的尊称合一
other_list = titanic.Title.value_counts().index[4:].tolist()
mapping = {}
for s in other_list:
    mapping[s] = 'Rare'
titanic['Title'] = titanic['Title'].map(lambda x: mapping[x] if x in mapping else x)

# 补全年龄
grouped = titanic.groupby(['Title'])
median = grouped.Age.median()


def new_age(cols):
    age = cols[0]
    title = cols[1]
    if pd.isnull(age):
        return median[title]
    return age


titanic.Age = titanic[['Age', 'Title']].apply(new_age, axis=1)

# 补全船舱号码
titanic.Cabin = titanic.Cabin.fillna('U')

# 补全登船信息
most_embarked = titanic.Embarked.value_counts().index[0]
titanic.Embarked = titanic.Embarked.fillna(most_embarked)

# 补全船票
titanic.Fare = titanic.Fare.fillna(titanic.Fare.median())

# 保留船舱类型
titanic['Cabin'] = titanic.Cabin.apply(lambda cabin: cabin[0])

# 把T类型转乘G(ABCDEF T?)
titanic['Cabin'].loc[titanic.Cabin == 'T'] = 'G'

# 提取家庭人数，由 Parch(父母/孩子)数量与 SibSp(兄弟姐妹/配偶)数量累加
titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1

# 舍去 Parch 和 SibSp 特征保留 FamilySize
titanic = titanic.drop(['SibSp', 'Parch'], axis=1)

# 年龄分层3级
titanic.Age = pd.cut(titanic.Age, 3, labels=False)

# 舍去名字
titanic = titanic.drop('Name', axis=1)

# 舍去票号
titanic = titanic.drop('Ticket', axis=1)

# 票价分为5级
titanic.Fare = pd.cut(titanic.Fare, 5, labels=False)

# 转换str 为 int
titanic.Sex = titanic.Sex.map({'male': 1, 'female': 0})
titanic.Cabin = titanic.Cabin.map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'U': 7})
titanic.Embarked = titanic.Embarked.map({'C': 0, 'Q': 1, 'S': 2})
titanic.Title = titanic.Title.map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

# TODO 构建模型

# 切割数据集
train = titanic[:len_train]
test = titanic[len_train:]

X_train = train.loc[:, 'Pclass':]
y_train = train['Survived']
X_test = test.loc[:, 'Pclass':]

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

print('log_reg:', log_reg.score(X_train, y_train))
print('svm_clf:', svm_clf.score(X_train, y_train))
print('tree_clf:', tree_clf.score(X_train, y_train))
print('xgb_clf:', xgb_clf.score(X_train, y_train))

# random_forest_clf = RandomForestClassifier(random_state=1)
# params_random_forest = [{'n_estimators': [10, 100], 'max_depth': [3, 6], 'criterion': ['gini', 'entropy']}]
# grid_search = GridSearchCV(estimator=random_forest_clf, param_grid=params_random_forest, scoring='accuracy', cv=2)
# scores_rf = cross_val_score(grid_search, X_train, y_train, scoring='accuracy', cv=5)
# model = grid_search.fit(X_train, y_train)
#
# print('scores_rf:', scores_rf)
# print('random_forest grid_search:', grid_search.score(X_train, y_train))

# 1. Bagging 算法实现
# 0.8691726623564537  [0.86179183 0.82700922 0.8855615  0.87700535 0.89449541]
rf_clf = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
rf_clf.fit(X_train, y_train)
# 2. Boosting 算法实现
# 0.8488710896477386  [0.8198946  0.82285903 0.87780749 0.84906417 0.87473017]
rb_clf = AdaBoostClassifier(random_state=1, n_estimators=100, learning_rate=1)
rb_clf.fit(X_train, y_train)

print('rf_clf:', rf_clf.score(X_train, y_train))
print('rb_clf:', rb_clf.score(X_train, y_train))

# 3. Voting
# 0.8695399796790022  [0.87259552 0.8370224  0.87433155 0.86885027 0.89490016]
model = VotingClassifier(
    estimators=[
        ('log_clf', LogisticRegression()),
        ('ab_clf', AdaBoostClassifier()),
        ('svm_clf', SVC(probability=True)),
        ('rf_clf', RandomForestClassifier()),
        ('gbdt_clf', GradientBoostingClassifier()),
        ('rb_clf', AdaBoostClassifier())
    ], voting='soft')  # , voting='hard')
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print('Voting :', scores.mean(), "--", scores)

# # 4. Stacking
# # 0.8713813265814722  [0.87747036 0.83886693 0.86590909 0.87085561 0.90380464]
# clfs = [
#     AdaBoostClassifier(),
#     SVC(probability=True),
#     AdaBoostClassifier(),
#     LogisticRegression(C=0.1, max_iter=100),
#     XGBClassifier(max_depth=6, n_estimators=100, num_round=5),
#     RandomForestClassifier(n_estimators=100, max_depth=6, oob_score=True),
#     GradientBoostingClassifier(learning_rate=0.3, max_depth=6, n_estimators=100)
# ]
#
# kf = KFold(n_splits=5, shuffle=True, random_state=1)
#
# # 创建零矩阵
# dataset_stacking_train = np.zeros((X_train.shape[0], len(clfs)))
# # dataset_stacking_label  = np.zeros((trainLabel.shape[0], len(clfs)))
#
# for j, clf in enumerate(clfs):
#     '''依次训练各个单模型'''
#     for i, (train, test) in enumerate(kf.split(y_train)):
#         '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
#         # print("Fold", i)
#         X_train, y_train, X_test, y_test = X_train[train], y_train[train], X_train[test], y_train[test]
#         clf.fit(X_train, y_train)
#         y_submission = clf.predict_proba(X_test)[:, 1]
#
#         # j 表示每一次的算法，而 test是交叉验证得到的每一行（也就是每一个算法把测试机和都预测了一遍）
#         dataset_stacking_train[test, j] = y_submission
#
# # 用建立第二层模型
# model = LogisticRegression(C=0.1, max_iter=100)
# model.fit(dataset_stacking_train, y_train)
#
# scores = cross_val_score(model, dataset_stacking_train, y_train, cv=5, scoring='roc_auc')
# print('Stacking :',scores.mean(), "--", scores)


# 5. Blending
# 0.8838950287185581 [0.87584416 0.91064935 0.89714286 0.85294118 0.8828976 ]
clfs = [
    AdaBoostClassifier(),
    SVC(probability=True),
    AdaBoostClassifier(),
    LogisticRegression(C=0.1, max_iter=100),
    XGBClassifier(max_depth=6, n_estimators=100, num_round=5),
    RandomForestClassifier(n_estimators=100, max_depth=6, oob_score=True),
    GradientBoostingClassifier(learning_rate=0.3, max_depth=6, n_estimators=100)
]
X_d1, X_d2, y_d1, y_d2 = train_test_split(X_train, y_train, test_size=0.5, random_state=2020)
dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
dataset_test = np.zeros((X_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((y_train.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    clf.fit(X_d1, y_d1)
    dataset_d1[:, j] = clf.predict_proba(X_d2)[:, 1]

    dataset_test[:, j] = dataset_d1.mean(1)

model = LogisticRegression(C=0.1, max_iter=100)
model.fit(dataset_d1, y_d2)

y_submission = model.predict_proba(dataset_test)[:, 1]
pred = model.predict(X_test)
print(pred)

scores = cross_val_score(model, dataset_d1, y_d2, cv=5, scoring='roc_auc')
print('Blending :', scores.mean(), "\n", scores)
