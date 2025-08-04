import torch

class LRScheduler():
    """
    학습률 스케줄러. 주어진 `patience` 에포크 동안 검증 손실이 감소하지 않으면 
    학습률을 주어진 `factor`만큼 감소.
    """
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5, verbose=False):
        """
        new_lr = old_lr * factor
        :param optimizer: 사용할 옵티마이저
        :param patience: 학습률을 업데이트하기 전에 기다릴 에포크 수
        :param min_lr: 학습률이 감소할 수 있는 최소 값
        :param factor: 학습률을 감소시킬 비율
        :param verbose: 학습률이 조정될 때 메시지를 출력할지 여부
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.verbose = verbose
        
        # ReduceLROnPlateau 스케줄러 사용
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=self.verbose
        )
        
    def __call__(self, val_loss):
        """
        검증 손실을 기준으로 학습률을 조정합니다.
        :param val_loss: 현재 검증 손실 값
        """
        self.lr_scheduler.step(val_loss)
