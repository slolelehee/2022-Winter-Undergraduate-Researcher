from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],test_size=0.25, random_state=4)

svm_model = svm.SVC(C=1,gamma=10)
svm_model.fit(X_train,y_train)
y_prediction = svm_model.predict(X_test)    
 
for i in range(0,len(y_prediction)):
    y_pred = y_prediction[i]
    print("{} : {}".format(X_test[i],iris_dataset['target_names'][y_pred]))

print("테스트 세트의 정확도 :",svm_model.score(X_test, y_test))