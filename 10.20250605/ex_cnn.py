import numpy as np
import matplotlib.pyplot as plt

# #체스보드 수직 엣지 구하기
# #샘플 이미지 생성
# sample_image = np.zeros((8,8))
# sample_image[1::2,::2] = 1
# sample_image[::2,1::2] = 1

# # plt.imshow(sample_image, cmap='gray')
# # plt.show()

# def convolution_2d(image, kernel, stride=1, padding=0):
#     #2d 컨볼루션 연산 구현
#     H, W = image.shape
#     K = kernel.shape[0]
#     print(f"K:{K}")
#     #패딩 적용
#     image = np.pad(image,padding, mode="constant")
#     H_padded, W_padded = image.shape
    
#     #출력 크기 계산
#     out_H = (H_padded - K) // stride + 1
#     out_W = (W_padded - K) // stride + 1

#     #출력 배열 초기화
#     output = np.zeros((out_H, out_W))

#     #컨볼루션 연산
#     for i in range(out_H):
#         for j in range(out_W):
#             #현재 위치
#             h_start = i * stride
#             h_end = h_start + K
#             w_start = j * stride
#             w_end = w_start + K
            
#             #해당 영역과 커널의 요소별 곱셈후 합
#             region = image[h_start:h_end, w_start:w_end]
#             output[i,j] = np.sum(region * kernel)
#     return output

# #수직 엣지 검출
# vertical_edge = np.array([
#     [-1,0,1],
#     [-1,0,1],
#     [-1,0,1]
# ])
# #수직 엣지 검출 적용
# padding =  (vertical_edge.shape[0] - 1) // 2
# vertical_edges = convolution_2d(sample_image, vertical_edge, padding=padding)

# print(f"원본 크기 : {sample_image.shape}")
# print(f"결과 크기 : {vertical_edges.shape}")

# #결과 시각화
# plt.figure(figsize=(10,5))

# plt.subplot(1,2,1)
# plt.title("original image")
# plt.imshow(sample_image, cmap='gray')

# plt.subplot(1,2,2)
# plt.title("vertical edges")
# plt.imshow(vertical_edges, cmap='gray')

# plt.tight_layout()
# plt.show()


# ########################################################

# #다중 채널 컨볼루션


# #다중 채널 컨볼루션
# def convolution_3d(image, filters,stride=1, padding=0):
#     #3d 컨볼루션
    
#     K, _, _, C_out = filters.shape
    
#     image = np.pad(image, ((padding,padding),(padding,padding),(0,0)))
#     H_padded, W_padded, C_in = image.shape
    
#     out_H = (H_padded - K) // stride + 1
#     out_W = (W_padded - K) // stride + 1    
#     output = np.zeros((out_H, out_W, C_out))
    
#     for f in range(C_out):
#         for i in range(out_H):
#             for j in range(out_W):
#                 h_start = i * stride
#                 h_end = h_start + K
#                 w_start = j * stride
#                 w_end = w_start + K
                
#                 #모든 입력 채널에서 연산후 합
#                 region = image[h_start:h_end, w_start:w_end, :]
#                 output[i,j,f] = np.sum(region * filters[:,:,:,f])
#     return output

# rgb_image = np.random.rand(32,32,3) # 32*32 컬러이미지
# filters = np.random.randn(3,3,3,16) # 16개의 3X3 필터

# #첫번째 필터의 첫번째 채널
# print(f"filters: {filters[:,:,0,0]}")

# padding =  (filters.shape[0] - 1) // 2
# feature_maps = convolution_3d(rgb_image, filters, padding=padding)
# print(f"입력 : {rgb_image.shape} -> 출력 : {feature_maps.shape}")

# #시각화
# plt.figure(figsize=(15,5))

# #원본 이미지
# plt.subplot(1,3,1)
# plt.title("original RGB image")
# plt.imshow(rgb_image)

# #첫번째 필터의 첫번째 채널
# plt.subplot(1,3,2)
# plt.title("first filter(1st channel)")
# plt.imshow(filters[:,:,0,0], cmap='gray')

# #컨볼루션 결과의 첫 번째 채널
# plt.subplot(1,3,3)
# plt.title("First feature map")
# plt.imshow(feature_maps[:,:,0], cmap='gray')

# plt.tight_layout()
# plt.show()


# ##########################################
# #풀링
# def max_pooling_2d(input_array, pool_size=(2,2), stride=2):
#     #2d 최대 풀링 구현
#     h, w  = input_array.shape
#     pool_h, pool_w = pool_size
    
#     #출력 크기 계산 패딩X
#     out_h = (h - pool_h) // stride + 1
#     out_w = (w - pool_w) // stride + 1
    
#     #출력 배열 초기화
#     output = np.zeros((out_h, out_w))
    
#     #풀링 연산
#     for i in range(out_h):
#         for j in range(out_w):
#             h_start = i * stride
#             h_end = h_start + pool_h
#             w_start = j * stride
#             w_end = w_start + pool_w
            
#             #현재 영역의 최대값 찾기
#             region = input_array[h_start:h_end, w_start:w_end]
#             output[i,j] = np.max(region)
            
#     return output


# from sklearn.datasets import load_sample_image

# image = load_sample_image("flower.jpg")

# gray_image = np.mean(image, axis=2)
# print(f"원본 이미지 크기 : {gray_image.shape}")

# #이미지에 풀링 적용
# pooled_image = max_pooling_2d(gray_image, pool_size=(4,4), stride=4)
# print(f"풀링 후 이미지 크기 : {pooled_image.shape}")

# #시각화
# fig, axes = plt.subplots(1,2, figsize=(12,5))

# axes[0].imshow(gray_image, cmap='gray')
# axes[0].set_title(f"Original Image{gray_image.shape}")
# axes[0].axis('off')

# axes[1].imshow(pooled_image, cmap='gray')
# axes[1].set_title(f"Max Pooling (4x4, stride=4) {pooled_image.shape}")
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()


##############################################

import tensorflow as tf
from tensorflow.keras import models, layers

def create_improved_cnn():
    model = models.Sequential([
        #block 1
        #기본 저수준 특징 추출
        layers.Conv2D(32, (3,3),padding = "same" ,activation='relu', input_shape=(32,32,3)),
        #배치 정규화
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3),padding = "same" ,activation='relu'),
        layers.MaxPooling2D((2,2)),
        #25% 뉴런 0으로 -> 과적합 방지
        layers.Dropout(0.25),
        
        
        #block 2
        #고수준 특징 추출
        #64개의 필터 사용
        layers.Conv2D(64, (3,3),padding = "same" ,activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3),padding = "same" ,activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        #block 3
        #더 복잡한 특징 추출
        layers.Conv2D(128, (3,3),padding = "same" ,activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3),padding = "same" ,activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),
        
        #Global Average Pooling
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
        
        
        
    ])

    return model

model = create_improved_cnn()
model.summary()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#전처리
x_train = x_train / 255.0
x_test = x_test / 255.0

#학습
model.fit(x_train, y_train, epochs=10, batch_size=64, 
          validation_split=0.2)

#평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
print(f"테스트 정확도 : {test_acc:.2%}")








