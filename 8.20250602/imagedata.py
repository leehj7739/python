#실습1
# import numpy as np
# import matplotlib.pyplot as plt

# simple_image = np.array([
#     [[255,0,0],     [0,255,0],     [0,0,255]],
#     [[255,255,0],   [0,255,255],   [255,0,255]],
#     [[0,0,0],       [128,128,128], [255,255,255]],
#   ], dtype=np.uint8)

# plt.rc("font", family="Malgun Gothic")
# plt.figure(figsize=(8,4))
# plt.imshow(simple_image)
# plt.title("3X3 픽셀 이미지")
# plt.axis("off")
# plt.show()

#실습2
# import cv2
# import matplotlib.pyplot as plt

# sample_image = cv2.imread("sample.jpg")

# #채널 분리 BGR
# blue_channel = sample_image[:,:,0]
# green_channel = sample_image[:,:,1]
# red_channel = sample_image[:,:,2]

# plt.figure(figsize=(15,3))
# plt.rc("font", family="Malgun Gothic")
# plt.subplot(1,4,1)
# plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
# plt.title("원본 이미지")
# plt.axis("off")

# plt.subplot(1,4,2)
# plt.imshow(red_channel, cmap="Reds")
# plt.title("Red 채널")
# plt.axis("off")

# plt.subplot(1,4,3)
# plt.imshow(green_channel, cmap="Greens")
# plt.title("Green 채널")
# plt.axis("off")

# plt.subplot(1,4,4)
# plt.imshow(blue_channel, cmap="Blues")
# plt.title("Blue 채널")
# plt.axis("off")

# plt.tight_layout()
# plt.show()


# #실습3 해상도
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# #이미지 로드
# image = cv2.imread("sample.jpg")

# #해상도 예제
# original_size = (400,400,3)
# original_image = np.random.randint(0,256, original_size, dtype=np.uint8)

# #다양한 해상도 버전 생성
# resolutions = [(400,400), (200,200), (100,100), (50,50)]

# plt.figure(figsize=(16,4))
# plt.rc("font", family="Malgun Gothic")

# for i, (width,height) in enumerate(resolutions):
#     #resized_image = cv2.resize(original_image,(width,height))
#     resized_image = cv2.resize(image,(width,height))
#     plt.subplot(1, 4, i+1)
#     plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
#     plt.title(f"해상도: {width}x{height} \n 픽셀 수: {width*height}")
#     plt.axis("off")
#     #파일 크기 계산(바이트)
#     file_size = resized_image.nbytes
#     print(f"해상도: {width}x{height} 파일 크기: {file_size/1024:.1f} KB")

# plt.tight_layout()
# plt.show()


# #실습 4 해상도와 품질의 관계 분석
# #해상도와 품질의 관계 분석
# def analyze_resolution_quality(image, target_size):
#     resized = cv2.resize(image, target_size)
#     restored = cv2.resize(resized, (image.shape[1], image.shape[0]))

#     #원본 이미지 크기로 복원
#     #빈공간은 픽셀 보간 방법으로 채워짐
    
#     mse = np.mean((image.astype(float) - restored.astype(float)) ** 2)

#     return resized, restored, mse

# #품질 손실 분석
# test_image = np.random.randint(0,256, (200,200,3), dtype=np.uint8)
# sizes = [(100,100), (50,50), (25,25)]

# for size in sizes:
#     _, restored, mse = analyze_resolution_quality(test_image, size)
#     print(f"해상도: {size} MSE: {mse:.2f}")


