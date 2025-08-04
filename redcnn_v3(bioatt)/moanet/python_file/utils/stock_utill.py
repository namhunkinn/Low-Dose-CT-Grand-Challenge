"""
Created on Sat July 17 2024

@author: YM
"""

from collections import defaultdict
import talib
import pandas as pd
import numpy as np
from typing import List
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler

class TechnicalIndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
   
        self.df = df

    def calculate_indicators(self, periods: List[int], scale: bool = False) -> pd.DataFrame:
        """
        다양한 기술 지표를 계산하는 함수

        Args:
            periods (list[int]): 지표 계산에 사용될 기간 리스트

        Returns:
            pd.DataFrame: 계산된 지표를 포함하는 데이터프레임
        """
        for T in periods:
            # ADOSC (Chaikin A/D Oscillator)
            self.df[f'ADOSC_{T}'] = talib.ADOSC(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'], fastperiod=T-3, slowperiod=T+4)

            # BBANDS (Bollinger Bands)
            
            UPPERBAND, MIDDLEBAND, LOWERBAND = talib.BBANDS(self.df['Close'], timeperiod=T, nbdevup=2, nbdevdn=2, matype=0)
            if scale:
                self.df[f'BBH_{T}'] = np.log(1 + (UPPERBAND - self.df['Close']) / self.df['Close'])
                self.df[f'BBL_{T}'] = np.log(1 + (self.df['Close'] - LOWERBAND) / self.df['Close'])
            else: 
                self.df[f'BBH_{T}'] = UPPERBAND
                self.df[f'BBL_{T}'] = LOWERBAND
                
            # CCI (Commodity Channel Index)
            self.df[f'CCI_{T}'] = talib.CCI(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=T)

            # CMO (Chande Momentum Oscillator)
            self.df[f'CMO_{T}'] = talib.CMO(self.df['Close'], timeperiod=T)

            # EMA (Exponential Moving Average)
            EMA = talib.EMA(self.df['Close'], timeperiod=T)
            if scale:                
                self.df[f'EMA_{T}'] = (EMA - EMA.shift(1)) / EMA.shift(1)
            else:
                self.df[f'EMA_{T}'] = EMA

            # Stochastic Fast
            FASTK, FASTD = talib.STOCHF(self.df['High'], self.df['Low'], self.df['Close'], fastk_period=T, fastd_matype=0)
            self.df[f'FASTD_{T}'] = FASTD

            # MA (Moving Average)
            MA = talib.MA(self.df['Close'], timeperiod=T)
            if scale:
                self.df[f'MA_{T}'] = (MA - MA.shift(1)) / MA.shift(1)
            else:
                self.df[f'MA_{T}'] = MA
                
            # MACD (Moving Average Convergence Divergence)
            MACD, MACDSIGNAL, MACDHIST = talib.MACD(self.df['Close'], fastperiod=T-5, slowperiod=T+10, signalperiod=T)
            self.df[f'MACD_{T}'] = MACD.shift(T)

            # MOM (Momentum)
            self.df[f'MOM_{T}'] = talib.MOM(self.df['Close'], timeperiod=T)

            # PPO (Percentage Price Oscillator)
            self.df[f'PPO_{T}'] = talib.PPO(self.df['Close'], fastperiod=T-5, slowperiod=T+5, matype=0)

            # ROC (Rate of Change)
            self.df[f'ROC_{T}'] = talib.ROC(self.df['Close'], timeperiod=T)

            # RSI (Relative Strength Index)
            self.df[f'RSI_{T}'] = talib.RSI(self.df['Close'], timeperiod=T)

            # WILLR (Williams %R)
            self.df[f'WILLR_{T}'] = talib.WILLR(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=T)

            # WMA (Weighted Moving Average)
            WMA = talib.WMA(self.df['Close'], timeperiod=T)
            if scale:
                self.df[f'WMA_{T}'] = (WMA - WMA.shift(1)) / WMA.shift(1)
            else: 
                self.df[f'WMA_{T}'] = WMA

            # RETURN
            self.df[f'RETURN_{T}'] = (self.df['Close'] - self.df['Close'].shift(T)) / self.df['Close'].shift(T)
               
        self.df['Label'] = (self.df['Close'].shift(-1) < self.df['Close']).astype(int)
        self.df = self.df.drop(columns=["High", "Low", "Volume"])
        self.df.dropna(inplace=True)
            

        return self.df
    

def make_sets(data_frame, test_start_date, test_period, val_period, train_period):
    sets = []  

    # DataFrame 인덱스에서 시간대 정보를 제거
    data_frame.index = data_frame.index.tz_localize(None)
    
    data_frame['date'] = data_frame.index

    for i, current_test_start_date in enumerate(pd.date_range(start=test_start_date, end=data_frame['date'].max(), freq=test_period), start=1):
        # 현재 테스트 세트의 종료 날짜
        current_test_end_date = current_test_start_date + test_period
        # 현재 검증 세트의 시작 날짜
        current_val_start_date = current_test_start_date - val_period
        # 현재 훈련 세트의 시작 날짜
        current_train_start_date = current_val_start_date - train_period

        test_set = data_frame[(data_frame['date'] >= current_test_start_date) & (data_frame['date'] < current_test_end_date)]
        val_set = data_frame[(data_frame['date'] >= current_val_start_date) & (data_frame['date'] < current_test_start_date)]
        train_set = data_frame[(data_frame['date'] >= current_train_start_date) & (data_frame['date'] < current_val_start_date)]
        
        sets.append((train_set, val_set, test_set))

        print(f"Set {i}:")
        print(f"Train set date range: {train_set['date'].min().strftime('%Y-%m-%d')} - {train_set['date'].max().strftime('%Y-%m-%d')}, sample count: {len(train_set)}")
        print(f"Val set date range: {val_set['date'].min().strftime('%Y-%m-%d')} - {val_set['date'].max().strftime('%Y-%m-%d')}, sample count: {len(val_set)}")
        print(f"Test set date range: {test_set['date'].min().strftime('%Y-%m-%d')} - {test_set['date'].max().strftime('%Y-%m-%d')}, sample count: {len(test_set)}")

    return sets  

def train_test(sets: List[tuple],epo=1,learning_rate=0.001,bs=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (train_set, val_set, test_set) in enumerate(sets):
        print(f"set {i + 1}:")
        print(f"Train set: {train_set['date'].dt.date.min()} to {train_set['date'].dt.date.max()} ({len(train_set)} samples)")
        print(f"Validation set: {val_set['date'].dt.date.min()} to {val_set['date'].dt.date.max()} ({len(val_set)} samples)")
        print(f"Test set: {test_set['date'].dt.date.min()} to {test_set['date'].dt.date.max()} ({len(test_set)} samples)")

        X_train_features = train_set.drop(columns=['date', 'Label', 'Open', 'Close'])
        X_val_features = val_set.drop(columns=['date', 'Label', 'Open', 'Close'])
        X_test_features = test_set.drop(columns=['date', 'Label', 'Open', 'Close'])

        # Robust scaling
        scaler = RobustScaler()
        #scaler = StandardScaler()

        X_train_features = scaler.fit_transform(X_train_features)
        X_val_features = scaler.transform(X_val_features)
        X_test_features = scaler.transform(X_test_features)

        # 이미지형태로 변환
        X_train = torch.tensor(X_train_features, dtype=torch.float32).view(-1, 1, 16, 16).to(device)
        X_val = torch.tensor(X_val_features, dtype=torch.float32).view(-1, 1, 16, 16).to(device)
        X_test = torch.tensor(X_test_features, dtype=torch.float32).view(-1, 1, 16, 16).to(device)

        y_train = torch.tensor(train_set['Label'].values, dtype=torch.long).to(device)
        y_val = torch.tensor(val_set['Label'].values, dtype=torch.long).to(device)
        y_test = torch.tensor(test_set['Label'].values, dtype=torch.long).to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=bs, shuffle=False)

        # 모델 초기화 및 학습
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epo)

        # Best model 불러오기
        model.load_state_dict(torch.load('best_model_set.pth'))

        # Validation 평가
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

         
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'Validation Accuracy: {accuracy:.4f} | Validation F1 Score: {f1:.4f}')

        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(len(cm))], columns=[f'Predicted {i}' for i in range(len(cm))])
        print('\nval Confusion Matrix:')
        print(cm_df)
        print()

        # Test 평가
        test_loss = 0
        all_preds = []
        all_labels = []

        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'Test Accuracy: {accuracy:.4f} | Test F1 Score: {f1:.4f}')

        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(len(cm))], columns=[f'Predicted {i}' for i in range(len(cm))])
        print('\nTest Confusion Matrix:')
        print(cm_df)
        print()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )


        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(in_features=16 * 6 * 6, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32, out_features=2)  
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
  

        
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, min_lr=0.00001)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        
        # 훈련 루프
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss.backward()
            optimizer.step()
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
       

        if (epoch + 1) % 50 == 0 or epoch + 1 == num_epochs:
             print(f'Epoch {epoch+1}/{num_epochs}')
             print(f'Train Loss: {train_loss/len(train_loader):.2f}, Train Accuracy: {accuracy:.2f}, Train F1 Score: {f1:.2f}')
        

        # 검증 루프
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        scheduler.step(val_loss)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), f'best_model_set.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break 
          