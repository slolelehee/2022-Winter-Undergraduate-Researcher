# 필요 모듈 import
from PIL import Image
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# 파이토치 모츌 중 딥러닝 모델을 설계할 때 필요한 함수를 모아 놓은 모듈
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms, datasets

# GPU가 사용이 가능하면 GPU로, 아니면 CPU로 DEVICE 설정
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')
  
# 학습시킬 데이터 가져오기
trans = transforms. Compose([transforms.Resize ( (100,100)),
                            # 이미지 데이터를 tensor 데이터로 변경
                            transforms.ToTensor(),
                            # 이미지 정규화
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                            ])

# 전체 트레이닝 데이터 셋을 여러 작은 그룹을 나누었을 때
# batch size는 하나의 소그룹에 속하는 데이터 수
BATCH_SIZE=15
# 딥러닝에서 epoch는 전체 트레이닝 셋이 신경망을 통과한 횟수
EPOCHS=10

# trainset 불러오기
trainset = torchvision.datasets.ImageFolder (root = "C:/Users/ehtl0/Desktop/python/animal_dataset/animal",
                                            transform = trans)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=BATCH_SIZE,
                                          shuffle=True, 
                                          num_workers=2)

# testset 불러오기
testset = torchvision.datasets.ImageFolder (root = "C:/Users/ehtl0/Desktop/python/MLP/test",
                                           transform = trans)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=BATCH_SIZE,
                                         shuffle=False, 
                                         num_workers=2)

classes=trainset.classes

trainset.__getitem__(0)

# 이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))

# 데이터셋 확인하기
for (X_train, y_train) in trainloader:
  print('X_train: ', X_train.size(), 'type: ', X_train.type())
  print('y_train: ', y_train.size(), 'type: ', y_train.type())
  break
  
# MLP 모델 설계하기
# 파이토치의 nn.Module 클래스 상속
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # full connected layer
    self.fc1 = nn.Linear(100*100*3,512)
    self.fc2 = nn.Linear(512,256)
    self.fc3 = nn.Linear(256,3)
  def forward(self,x):
    # 3차원 데이터를 flatten
    x = x.view(-1, 100*100*3)
    x = self.fc1(x)
    x = F.sigmoid(x)
    x = self.fc2(x)
    x = F.sigmoid(x)
    x = self.fc3(x)
    x = F.log_softmax(x, dim=1)
    return x
    
# MLP 모델에서 사용될 DEVICE 할당
model = Net().to(DEVICE)
# optimiaer 정의
# SGD 알고리즘을 이용해 파라미터 업데이트
# lr : learning rate, momentum : optimizer의 관성을 나타냄
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()

print(model)

# MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의
def train(model, train_loader, optimizer, log_interval):
  model.train()
  for batch_idx,(image,label) in enumerate(train_loader):
    image = image.to(DEVICE)
    label = label.to(DEVICE)
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print("Train Epoch : {} [{}/{}({:.0f}%)] \tTrain Loss: {:.6f}".format(Epoch, batch_idx * len(image),
            len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            
# 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의
def evaluate(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0 

  with torch.no_grad():
    for image, label in test_loader:
      image = image.to(DEVICE)
      label = label.to(DEVICE)
      output = model(image)
      test_loss += criterion(output, label).item()
      prediction = output.max(1, keepdim = True)[1]
      correct += prediction.eq(label.view_as(prediction)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, test_accuracy
  
# MLP 학습
for Epoch in range(1, EPOCHS+1):
  train(model, trainloader,optimizer,log_interval = 200)
  test_loss, test_accuracy = evaluate(model,testloader)
  print("\n[EPOCH: {}], \tTest Loss : {:.4f}, \tTest Accuracy: {:.2f} %\n"
        .format(Epoch, test_loss, test_accuracy))
