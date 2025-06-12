import cv2
import numpy as np
import matplotlib.pyplot as plt


print(f"opnecv 버전: {cv2.__version__}")

plt.rc("font",family="Malgun Gothic")

#그라디언트 기본 이미지 생성
def create_sample_image():
    height, width = 300,400
    image = np.zeros((height,width,3), dtype=np.uint8)
    
    #기하학적 도형 추가
    cv2.rectangle(image, (50,50), (150,150), (255,255,255), -1)
    cv2.circle(image, (300,200), 50, (0,255,255), -1)
    cv2.line(image, (200,100), (350,250), (255,0,255), 3)
    
    return image

sample = create_sample_image()
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
plt.title("OpenCV로 생성한 샘플이미지")
plt.axis("off")
plt.show()


