import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#간단한 집값 예측 머신러닝 모델
def house_price_prediction():
    #가상의 집 데이터 생성
    np.random.seed(42)
    
    #house_size : 집 크기 / 평균, 표준편차, 갯수
    house_size = np.random.normal(100, 30, 1000)
    print(house_size)

    #house_price : 집 가격 
    #크기가 클 수록 가격이 높아지는 관계 + 노이즈
    house_prices = house_size * 50 + np.random.normal(0, 500, 1000) + 2000
    print(house_prices)
    
    #데이터 전처리
    x = house_size.reshape(-1, 1) # 2d 배열로 변환
    y = house_prices
    
    # 훈련용/테스트용 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
        )
    
    #머신러닝 모델 생성 및 훈련
    model = LinearRegression()
    #모델 훈련
    model.fit(x_train, y_train)

    #예측
    y_pred = model.predict(x_test)
    
    #성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"평균 제곱 오차: {mse:.2f}")
    print(f"결정 계수 (R^2): {r2:.2f}")
    print(f"모델계수 (기울기): {model.coef_[0]:.2f}")
    print(f"모델 절편 : {model.intercept_:.2f}")
    
    #새로운 집 크기에 대한 예측
    new_house_sizes = [80, 120, 150]
    for size in new_house_sizes:
        predicted_price = model.predict([[size]])[0]
        print(f"{size} 평 집의 예상 가격: {predicted_price:.2f} 만만원")
    
    return model, x_test, y_test, y_pred
    

model, x_test, y_test, y_pred = house_price_prediction()

# 예측 결과 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10, 5))
plt.scatter(x_test, y_test, color='blue', label='실제 집 가격')
plt.plot(x_test, y_pred, color='red', label='예측 집 가격')
plt.xlabel('집 크기 (평)')
plt.ylabel('집 가격 (만원)')
plt.title('집 크기와 집 가격의 관계')
plt.legend()
plt.show()








