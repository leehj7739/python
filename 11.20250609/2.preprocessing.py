import cv2
import numpy as np
import matplotlib.pyplot as plt

class OCRPreprocesor:
    def __init__(self, image):
        pass
    
    #그레이스케일
    def convert_to_grayscale(self,image):
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return gray
    
    #이진화 처리리
    def apply_threshold(self,image, method="adaptive"):
        
        gray = self.convert_to_grayscale(image)
        
        if method == "simple":
            _, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        elif method == "adaptive":
            
            thresh = cv2.adaptiveThreshold(
                gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,11,2
                )
        elif method == "otsu":
            _, thresh = cv2.threshold(
                gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        return thresh
    
    #노이즈 제거
    def remove_noise(self, image):
        #필터
        kernel = np.ones((3,3),np.uint8)
        
        #침식, 팽창으로 작은 노이즈 점 제거
        opening = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
        #가우시안 블러로 미세한 노이즈 제거
        denoised = cv2.GaussianBlur(opening, (3,3), 0)
        
        return denoised         
    
    #기울기 보정
    def correct_skew(self, image):
        #canny 에지 검출
        edges = cv2.Canny(image,50,150,apertureSize=3)
        #허프 변환으로 직선 검출
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:            
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)
                
            median_angle = np.median(angles)
            
            (h, w) = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            rotated = cv2.warpAffine(image,M,(w,h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        
        return image
    
    #이미지 리사이즈
    def resize_image(self, image, target_height=800):
        h, w = image.shape[:2]
        
        if h > target_height:
            scale = target_height / h
            new_w = int(w * scale)
            
            resized = cv2.resize(image, (new_w,target_height),
                                 interpolation=cv2.INTER_CUBIC)
        else:
            resized = image
            
        return resized
    
    def visualize_preprocessing(self, steps, step_names):
        plt.rc("font",family="Malgun Gothic")
        fig, axes = plt.subplots(2,3 ,figsize=(15,10))
        
        axes = axes.ravel()
        
        for i, (step,name) in enumerate(zip(steps,step_names)):
            if i < len(axes):
                if len(step.shape) == 3:
                    axes[i].imshow(cv2.cvtColor(step,cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(step,cmap="gray")
                    
                axes[i].set_title(name)
                axes[i].axis("off")
                
        plt.tight_layout()
        plt.show()
            
    def process_pipeline(self,image, visualize = True):
        
        steps = []
        step_names = []
        
        #원본 이미지 보존
        steps.append(image.copy())
        step_names.append("원본 이미지")
        
        #1. 그레이 스케일 변환
        gray = self.convert_to_grayscale(image)
        steps.append(gray)
        step_names.append("그레이 스케일")
        
        #2. OCR 최적화를 위한 크기 조정
        resized = self.resize_image(gray)
        steps.append(resized)
        step_names.append("크기 조정")
        
        #3. 적응적 이진화 처리
        thresh = self.apply_threshold(resized, method="adaptive")
        steps.append(thresh)
        step_names.append("이진화")
        
        #4.모폴로지 연산으로 노이즈 제거
        denoised = self.remove_noise(thresh)
        steps.append(denoised)
        step_names.append("노이즈 제거")
        
        #5. 허프 변환 기반 기울기 자동 보정
        corrected = self.correct_skew(denoised)
        steps.append(corrected)
        step_names.append("기울기 보정")
        
        if visualize:
            self.visualize_preprocessing(steps,step_names)
            
        return corrected
        
def create_noisy_sample_image():
    image = np.ones((300,800,3),dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,"Noisy OCR Test Image",(50,100),font,1.5,(0,0,0),2)
    cv2.putText(image,"Preprocessing improves accuracy",(50,150),font,1.0,(0,0,0),2)
    cv2.putText(image,"Machine Learning & AI",(50,200),font,1.0,(0,0,0),2)
    
    noise = np.random.normal(0,50,image.shape).astype(np.uint8)
    noisy_image = cv2.add(image,noise)
    
    h, w = noisy_image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, 5, 1)
    skewed_image = cv2.warpAffine(noisy_image,M,(w,h))
    return skewed_image
        
def preprocessing_example():
    image = create_noisy_sample_image()
    preprocessor = OCRPreprocesor(image)
    
    processed_image = preprocessor.process_pipeline(image,visualize=True )
    
    try:
        import pytesseract
        original_text = pytesseract.image_to_string(image)
        processed_text = pytesseract.image_to_string(processed_image)
        
        print("원본 텍스트 : ",original_text)
        print("전처리 후 텍스트 : ",processed_text)
        
    except ImportError:
        print("pytesseract 라이브러리가 설치되지 않았습니다.")
        
    return processed_image


preprocessing_example()
        
            