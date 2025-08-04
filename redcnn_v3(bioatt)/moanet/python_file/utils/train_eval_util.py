import torch


def train(model, train_loader, optimizer, criterion, device):

    # 모델을 학습 모드로 설정
    model.train()
    # loss 누적값 초기화
    running_loss = 0.0

    # 정확하게 예측된 샘플 수를 초기화
    correct_preds = 0
    # 모든 예측과 실제 레이블을 저장할 리스트를 초기화
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader: # 로더에서 배치 사이즈만큼 데이터와 레이블 랜덤 추출
        inputs, labels = inputs.to(device), labels.to(device) # device로 옮기기

        # 기울기 초기화
        optimizer.zero_grad()

        outputs = model(inputs) # 모델에 인풋데이터 입력하여 아웃풋 추출

        loss = criterion(outputs, labels) # 아웃풋과 레이블과의 loss 계산

        loss.backward() # 파라미터에 대한 loss 기울기 계산
        optimizer.step() # 파라미터 업데이트

        # loss 누적
        running_loss += loss.item() * inputs.size(0)

        # 예측된 클래스를 가져와 정확하게 예측된 샘플 수를 누적
        _, predicted = torch.max(outputs, 1)
        correct_preds += torch.sum(predicted == labels).item()

        # 예측 결과와 실제 레이블을 리스트에 추가
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 전체 데이터셋에 대한 평균 손실을 계산
    avg_loss = running_loss / len(train_loader.dataset)
    # 정확도를 계산
    accuracy = correct_preds / len(train_loader.dataset)
    # 평균 손실, 정확도, 예측된 값들, 실제 레이블들을 반환
    return avg_loss, accuracy, all_preds, all_labels

# 평가 함수 정의
def evaluate(model, loader, criterion, device):
    # 모델을 평가 모드로 설정
    model.eval()
    running_loss = 0.0

    # 정확하게 예측된 샘플 수를 초기화
    correct_preds = 0
    # 모든 예측과 실제 레이블을 저장할 리스트를 초기화
    all_preds = []
    all_labels = []

    # 기울기를 계산하지 않도록 torch.no_grad() 내에서 연산 수행
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # 예측된 클래스를 가져와 정확하게 예측된 샘플 수를 누적
            _, predicted = torch.max(outputs, 1)
            correct_preds += torch.sum(predicted == labels).item()

            # 예측 결과와 실제 레이블을 리스트에 추가
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 전체 데이터셋에 대한 평균 손실을 계산
    avg_loss = running_loss / len(loader.dataset)
    # 정확도를 계산
    accuracy = correct_preds / len(loader.dataset)
    # 평균 손실, 정확도, 예측된 값들, 실제 레이블들을 반환
    return avg_loss, accuracy, all_preds, all_labels
