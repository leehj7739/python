def mini_mnist():
    # 8x8 숫자 이미지 분류 ( sklearn digits 데이터셋 사용)
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    
    #데이터 로드
    digits = load_digits()
    X, y = digits.data, digits.target
    X = X / 16.0 # 정규화 (0-16 -> 0-1)
    
    #원-핫 인코딩
    y_onehot = np.eye(10)[y]
    
    #훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
    print("미니 MNIST 과제")
    print(f"훈련 데이터 : {X_train.shape}")
    print(f"테스트 데이터 : {X_test.shape}")
    print(f"클래스 수 : 10개 (0~9 숫자자)")
    
    #TODO: 다중 클래스 분류 신경망 구현
    #1. 입력층 : 64개 (8X8 이미지)
    #2. 은닉층 : 적절한 크기
    #3. 출력층 : 10개 (0~9 숫자)
    #4. softmax 활성화 함수
    #5. Cross-Entropy 손실 함수
    
    class DigitClassifier:
        def __init__(self, input_size=64, hidden_size=200, output_size=10):
            
            #가중치, 편향 초기화
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/(input_size + hidden_size))
            self.b1 = np.zeros((1, hidden_size))
            
            self.W2 = np.random.randn(hidden_size, output_size)  * np.sqrt(2.0/(hidden_size + output_size))
            self.b2 = np.zeros((1, output_size))
            
            #중간 계산값 저장용
            self.z1 = None
            self.a1 = None
            self.z2 = None
            self.a2 = None
        
        def relu(self, z):
            return np.maximum(0, z)
        
        def relu_derivative(self, z):
            return np.where(z > 0, 1, 0)
            
        def softmax(self, x):
            e_z = np.exp(x-np.max(x, axis=1, keepdims=True))
            return e_z / np.sum(e_z, axis=1, keepdims=True)
        
        def cross_entropy_loss(self, y_pred, y_true):
            epsilon = 1e-15  # 수치 안정성을 위한 작은 값
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        def forward(self, X):
            #순전파            
            #입력 -> 은닉
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = self.relu(self.z1)
            
            #은닉 -> 출력
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.softmax(self.z2)
            
            return self.a2
        
        def backward(self, X, y, learning_rate = 0.01):
            #역전파
            #샘플 갯수
            m = X.shape[0]
            #출력층 오차 계산
            dz2 = self.a2 - y
            dW2 = (1/m) * np.dot(self.a1.T, dz2)
            db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
            
            #은닉층으로 오차 전파 
            dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
            
            #은닉층 w, b
            dW1 = (1/m) * np.dot(X.T, dz1)
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
            
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            
            return dW1, db1, dW2, db2
        
        def train(self, X, y, epochs=1000, learning_rate=0.01):
            #전체 학습 과정
            losses = []
            for epoch in range(epochs):
                
                # 학습률 감소
                current_lr = learning_rate / (1 + 0.001 * epoch)            
            
                #순전파
                output = self.forward(X)
                
                #손실 계산
                loss = self.cross_entropy_loss(output, y)
                losses.append(loss)
                
                #역전파
                self.backward(X,y, current_lr)
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch} : 손실 {loss}")
                    
            return losses
        
        
        # def predict(self, X):            
        #     predictions = self.forward(X)
        #     print(f"예측값 : {predictions.flatten()}")
        #     return np.argmax(predictions, axis=1)
            
        
        def accuracy(self, X, y):
            predictions = self.forward(X)
            predicted_classes = np.argmax(predictions, axis=1)
            print(f"예측값 : {predicted_classes}")
            true_classes = np.argmax(y, axis=1)
            print(f"정답값 : {true_classes}")
            return np.mean(predicted_classes == true_classes)
            
        def accuracy_per_class(self, X, y):
            predictions = self.forward(X)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
            accuracy_per_class = {}
            for i in range(10):
                class_mask = true_classes == i
                if np.sum(class_mask) > 0:
                    accuracy_per_class[i] = f"{np.mean(predicted_classes[class_mask] == i) *100:.2f}%"
            return accuracy_per_class
    
    nn = DigitClassifier()
    print(f"== 학습 전 ==")
    # nn.predict(X)
    print(f"학습전 정확도 : {nn.accuracy(X, y_onehot) * 100:.2f}%")
    losses =nn.train(X_train, y_train)
    print(f"== 학습 완료 ==")
    # nn.predict(X)
    print(f"훈련 정확도 : {nn.accuracy(X_train, y_train) * 100:.2f}%")
    print(f"테스트 정확도 : {nn.accuracy(X_test, y_test) * 100:.2f}%")
    print(f"각 숫자별 정확도 : {nn.accuracy_per_class(X_test, y_test)}%")
    
    
    plt.figure(figsize=(12, 8))
    plt.rc('font', family='Malgun Gothic')
    # 1. 손실 그래프
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title("학습 손실 변화")
    plt.xlabel("반복 횟수")
    plt.ylabel("손실 cross-entropy")
    plt.grid(True)

    #2. 가중치 변화 시각화
    plt.subplot(2, 2, 2)
    plt.bar(range(len(nn.W1.flatten())), nn.W1.flatten())
    plt.title("은닉층 가중치(학습후)")
    plt.xlabel("가중치 인덱스")
    plt.ylabel("가중치 값")
    plt.grid(True)

        #3. 예측결과 비교
    plt.subplot(2, 2, 3)
    x_pos = np.arange(10)  # 0부터 9까지의 숫자

    #3. 예측결과 비교
    plt.subplot(2, 2, 3)
    x_pos = np.arange(10)  # 0부터 9까지의 숫자

    # 각 숫자별 정확도 계산
    accuracy_dict = nn.accuracy_per_class(X_test, y_test)
    accuracies = [float(accuracy_dict[i].strip('%')) for i in range(10)]

    plt.bar(x_pos, accuracies, width=0.6)
    plt.title("각 숫자별 정확도")
    plt.xlabel("숫자")
    plt.ylabel("정확도 (%)")
    plt.xticks(x_pos, np.arange(10))
    plt.grid(True)
    plt.ylim(0, 100)  # y축 범위를 0-100%로 설정

    # 각 막대 위에 정확도 값 표시
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')

    plt.tight_layout()
    plt.show()
    
        
mini_mnist()
#정확도 상승을 위해 은닉층 크기 조정
#relu 활성화 함수 사용

