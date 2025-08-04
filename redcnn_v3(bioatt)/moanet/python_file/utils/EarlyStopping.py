class EarlyStopping:
    """
    조기 종료(Early Stopping) 클래스: 특정 에폭 동안 손실(loss)이 개선되지 않으면 훈련 중단
    """

    def __init__(self, patience=3, min_delta=0):
        """
        초기화 함수: patience, min_delta 값을 설정하고 카운터, best_loss를 초기화

        min_delta (float): 새로운 손실이 이전 손실보다 개선되었다고 간주하기 위한 최소 차이
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        """
         검증 손실(val_loss)을 받아 조기 종료 여부를 결정

        Returns:
            bool: 조기 종료해야 하는 경우 True, 계속 훈련해야 하는 경우 False
        """
        if self.best_loss is None:  # 첫 번째 에폭인 경우 best_loss 설정
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:  # 손실이 충분히 개선된 경우
            self.best_loss = val_loss
            self.counter = 0  # 카운터 초기화

        else:  # 손실이 개선되지 않은 경우
            self.counter += 1

            if self.counter >= self.patience:  # patience 만큼 손실이 개선되지 않으면
                print("[INFO] Early stopping")  # 조기 종료 메시지 출력
                return True  # 조기 종료 신호 반환

        return False  # 계속 훈련 신호 반환
