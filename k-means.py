# pandas 패키지 임포트
import pandas as pd

# read_excel() 함수를 이용하여 파일 불러오기
data = pd.read_excel('CustomerDataSet.xls')

# 데이터 몇 행만 보기
data.head()
# 필요 패키지 불러오기 (KMeans, matplotlib, preprocessing)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 원본 데이터를 복사해서 전처리하기 (원본 데이터를 가지고 바로 전처리하지 않는다)
processed_data = data.copy()

# 데이터 전처리 - 정규화를 위한 작업
scaler = preprocessing.MinMaxScaler()
processed_data[['ItemsBought', 'ItemsReturned']] = scaler.fit_transform(processed_data[['ItemsBought', 'ItemsReturned']])


plt.figure(figsize = (10, 6))
for i in range(1, 7):
    estimator = KMeans(n_clusters = i)
    ids = estimator.fit_predict(processed_data[['ItemsBought', 'ItemsReturned']])
    plt.subplot(3, 2, i)
    plt.tight_layout()
        # 서브플롯의 라벨링
    plt.title("K value = {}".format(i))
    plt.xlabel('ItemsBought')
    plt.ylabel('ItemsReturned')
        # 클러스터링 그리기
    plt.scatter(processed_data['ItemsBought'], processed_data['ItemsReturned'], c=ids)  
plt.show()