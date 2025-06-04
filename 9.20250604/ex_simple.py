import numpy as np

def simple_neuron_chain_rule():
    print("\n 단일 뉴런 연쇄법칙")
    
    #신경망 구조 : x -> z -> σ -> L
    # z = w*x  + b (선형변환)
    # a = σ(z) (시그모이드 활성화)
    # L = (a-t)**2 (평균제곱오차 손실)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)
    
    #실제 값들
    x = 1.0
    w = 0.5
    b = 0.2
    t = 0.8
    
    #순전파 계산
    print("순전파 :")
    z = w*x + b
    a = sigmoid(z)
    L = (a-t)**2
    
    print(f" z = {w}x{x} + b = {z:.3f}")
    print(f" a = σ({z:.3f}) = {a:.3f}")
    print(f" L = ({a:.3f}-{t})^2 = {L:.3f}")
    
    #역전파
    print("\n 역전파 dL/dW 계산:")
    #각 단계별 미분
    dl_da = 2*(a-t)
    da_dz = sigmoid_derivative(z)
    dz_dw = x
    
    print(f" dL/da = 2({a}-{t}) = {dl_da:.3f}")
    print(f" da/dz = σ'(z) = {da_dz:.3f}")
    print(f" dz/dw = x = {dz_dw}")
    
    #연쇄법칙 적용
    dL_dw = dl_da * da_dz * dz_dw
    
    #가중치 업데이트
    learning_rate = 0.1
    w_new = w - learning_rate * dL_dw
    
    return w_new


new_weight = simple_neuron_chain_rule()
print(f"\n 업데이트된 가중치 : w = {new_weight:.3f}")
    
