import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

def mnist_deep_learning():
    #데이터 로드
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #x_train_one 데이터 파일로 저장
    x_train_one = X_train[0]
    plt.imsave("x_train_one.png", x_train_one, cmap="gray")
    
    #데이터 정규화
    X_train = X_train.reshape(-1,28,28,1).astype("float32") / 255.0
    X_test = X_test.reshape(-1,28,28,1).astype("float32") / 255.0
    
    #레이블 원-핫 인코딩
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    #딥러닝 모델 구성 CNN
    model = Sequential([
        #첫번째 컨볼루션 블록
        Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        
        #두번째 컨볼루션 블록
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        
        #세번째 컨볼루션 블록
        Conv2D(64, (3,3), activation="relu"),
        
        #완전 연결층
        Flatten(),
        
        Dense(64, activation="relu"),
        
        Dropout(0.5), # 과적합 방지
        Dense(10, activation="softmax") # 10개 숫자 분류류
        
    ])
    
    #모델 컴파일
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    #모델 훈련
    print("모델 훈련을 시작합니다...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=5,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    #모델 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"최종 평가 정확도: {test_accuracy:.4f}")
    
    #예축 예시
    sample_index = np.random.randint(0, len(X_test), 100)
    predictions = model.predict(X_test[sample_index])
    
    #예측 결과
    print("예측 결과:")
    for i, idx in enumerate(sample_index):
        #예측 결과중 가장 높은 값의 인덱스
        predicted_digit = np.argmax(predictions[i])
        #실제 레이블중 가장 높은 값의 인덱스
        actual_digit = np.argmax(y_test[idx])
        #예측 결과중 가장 높은 값
        confidence = np.max(predictions[i]) * 100
        print(f"샘플 {i+1}: 예측={predicted_digit}, 실제={actual_digit}, 신뢰도={confidence:.1f}%")
    
    
    # 모델 저장 시 include_optimizer=True 추가
    model.save('mnist_model.h5', save_format='h5', include_optimizer=True)
    print("모델이 저장되었습니다.")
    
    
    return model, history


try:
    model = tf.keras.models.load_model('mnist_model.h5')
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("저장된 모델을 불러왔습니다.")
except:
    print("저장된 모델이 없어 새로 학습을 시작합니다.")
    model, history = mnist_deep_learning()
    
    
    


# 테스트 데이터 로드
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_test = to_categorical(y_test, 10)    

# 샘플 예측
sample_indices = np.random.randint(0, len(X_test), 100)
predictions = model.predict(X_test[sample_indices])
for i, idx in enumerate(sample_indices):
    #예측 결과중 가장 높은 값의 인덱스
    predicted_digit = np.argmax(predictions[i])
    #실제 레이블중 가장 높은 값의 인덱스
    actual_digit = np.argmax(y_test[idx])
    #예측 결과중 가장 높은 값
    confidence = np.max(predictions[i]) * 100
    print(f"샘플 {i+1}: 예측={predicted_digit}, 실제={actual_digit}, 신뢰도={confidence:.1f}%")

