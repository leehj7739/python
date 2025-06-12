import numpy as np
import matplotlib.pyplot as plt

#샘플 데이터
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

#정답
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

#시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class simpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #가중치 초기화
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        print(f"W1 형태: {self.W1.shape}")
        self.b1 = np.zeros((1, hidden_size))
        print(f"b1 형태: {self.b1.shape}")
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        print(f"W2 형태: {self.W2.shape}")
        self.b2 = np.zeros((1, output_size))
        print(f"b2 형태: {self.b2.shape}")
        
        #중간 계산값 저장용 (역전파에서 사용)
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
    def forward(self,X):
        print("=== 순전파 과정 ===")
        #입력층 -> 은닉층
        self.z1 = np.dot(X,self.W1) + self.b1
        print(f"z1 형태: {self.z1.shape}")
        self.a1 = sigmoid(self.z1)
        print(f"a1 형태: {self.a1.shape}")
        
        #은닉층 -> 출력층
        self.z2 = np.dot(self.a1,self.W2) + self.b2
        print(f"z2 형태: {self.z2.shape}")
        self.a2 = sigmoid(self.z2)
        print(f"a2 형태: {self.a2.shape}")
        
        return self.a2
    
    def backward(self,X,y,learning_rate):
        print("\n=== 역전파 과정 ===")
        #샘플 개수
        m = X.shape[0]
        print(f"샘플 개수 m : {m}")
        
        #1단계 출력층 오차 계산
        dz2 = self.a2 - y
        print(f"1단계 출력층 오차 dz2 : {dz2.flatten()}")
        
        # 2단계 출력층 가중치와 현향의 기울기
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        print(f"2단계 출력층 가중치 기울기 dW2 형태 : {dW2.shape}")
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        print(f"2단계 출력층 편편향 기울기 db2 형태 : {db2.shape}")
        
        #3단계 : 은닉층으로 오차 전파
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.a1)
        print(f"3단계 은닉층 오차 dz1 형태 : {dz1.shape}")

        #4단계 : 은닉층 가중치와 편향의 기울기
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        print(f"4단계 은닉층 가중치 기울기 dW1 형태 : {dW1.shape}")

        #5단계: 가중치 업데이트(경사하강법)
        print(f"5단계 - 가중치 업데이트 (학습률 : {learning_rate})")
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return dW1, db1, dW2, db2
        
        
            
    def train(self, X, y, epochs, learning_rate=0.1):
        # 전체 학습과정
        losses = []
        
        for epoch in range(epochs):
            #순전파
            output = self.forward(X)
                        
            #손실 계산 (평균 제곱 오차)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            #역전파
            self.backward(X,y,learning_rate)
            
            if epoch % 100 == 0:
                print(f"에포크크 {epoch} - 손실 : {loss}")
                
        return losses
    
nn = simpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

#학습전 예측
print("학습전 예측:")
before_output = nn.forward(X)
print(f"예측 결과 : {before_output.flatten()}")
print(f"정답값 : {y.flatten()}")

#학습 실행
losses = nn.train(X,y,epochs=1000,learning_rate=1.0)

#학습후 예측
print("\n학습후 예측:")
after_output = nn.forward(X)
print(f"예측 결과 : {after_output.flatten()}")
print(f"정답값 : {y.flatten()}")


plt.figure(figsize=(12,8))
plt.rc("font",family="Malgun Gothic")
#1. 손실그래프
plt.subplot(2,2,1)
plt.plot(losses)
plt.title("학습 손실 변화")
plt.xlabel("에포크")
plt.ylabel("손실(MSE)")
plt.grid(True)

#2. 가중치 변화 시각화
plt.subplot(2,2,2)
plt.bar(range(len(nn.W1.flatten())), nn.W1.flatten())
plt.title("은닉층 가중치 (학습후)")
plt.xlabel("가중치 인덱스")
plt.ylabel("가중치 값")

#3. 예측 결과 비교
plt.subplot(2,2,3)
x_pos = np.arange(len(y))
plt.bar(x_pos -0.2, y.flatten(), 0.4, label="정답", alpha=0.7)
plt.bar(x_pos +0.2, after_output.flatten(), 0.4, label="예측", alpha=0.7)
plt.title("예측 결과 비교")
plt.xlabel("샘플")
plt.ylabel("값")
plt.legend()
plt.xticks(x_pos, ["[0,0]","[0,1]","[1,0]","[1,1]"])

plt.subplot(2,2,4)
plt.text(0.1,0.7, "입력층\n(2개)", ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightblue"))
plt.text(0.5,0.7, "은닉층\n(4개)", ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightgreen"))
plt.text(0.9,0.7, "출력층\n(1개)", ha="center", va="center",
         bbox=dict(boxstyle="round", facecolor="lightcoral"))

#화살표 그리기
plt.arrow(0.2,0.7,0.2, 0, head_width=0.02, head_length=0.02, fc="black")
plt.arrow(0.5,0.7,0.2, 0, head_width=0.02, head_length=0.02, fc="black")
         
#화살표 그리기
plt.xlim(0,1)
plt.ylim(0.5,0.9)
plt.title("신경망 구조")
plt.axis("off")

plt.tight_layout()
plt.show()

print("학습 완료")
print(f"최종 손실 : {losses[-1]:.4f}")
print(f"XOR 문제 해결 정확도 : {np.mean(np.abs(after_output - y) < 0.1 ) * 100 :.1f}%")
