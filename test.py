from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],test_size=0.25, random_state=5)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
import numpy as np 

xtrainmean=np.mean(X_train, axis=0)
xtrainstd = np.std(X_train, axis=0)
xtrain_scaled = (X_train - xtrainmean) / xtrainstd


xtestmean=np.mean(X_test, axis=0)
xteststd = np.std(X_test, axis=0)
xtest_scaled = (X_test - xtestmean) / xteststd

knn.fit(xtrain_scaled, y_train)
y_prediction = knn.predict(xtest_scaled)
#X_test(data)에 대하여 품종예측
 
for i in range(0,len(y_prediction)):
    y_pred = y_prediction[i]
    print("{} : {}".format(X_test[i],iris_dataset['target_names'][y_pred]))

print("테스트 세트의 정확도 :",knn.score(X_test, y_test))