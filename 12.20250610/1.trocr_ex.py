from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
import requests
import torch
import numpy as np

class TrOCRSystem:
    def __init__(self, model_name ="microsoft/trocr-base-printed"):
        
        print(f"TrOCR 모델 로딩중 : {model_name}")
        #프로세서와 모델 로드
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        #프로세서: 이미지 전처리 및 토큰화
        #모델 : 트랜스포머 기반 모델
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"모델 로딩 완료 (device: {self.device})")
        
    def extract_text(self, image, return_confidence=False):
        #이미지에서 텍스트 추출
        #다양한 입력형태 지원원
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            if image.startswith("http"):
                image = Image.open(requests.get(image, stream=True).raw)
            else:
                image = Image.open(image)
    
        #이미지 전처리 - TrOCR 모델에 맞는 형태로 변환            
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        #연산장치 지정
        
        #텍스트 추출 인식
        with torch.no_grad():
            
            if return_confidence:
                outputs = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=256
                )
                generated_ids = outputs.sequences
                token_scores = outputs.scores
            else:
                generated_ids = self.model.generate(pixel_values)
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if return_confidence:
                
                if token_scores:
                    
                    token_probs = []
                    for score in token_scores:
                        
                        probs = torch.softmax(score, dim=-1)
                        max_prob = torch.max(probs).item()
                        token_probs.append(max_prob)
                        
                        confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
                        
                        
                        print(f"신뢰도 계산 상세 :")
                        print(f"토큰별 확률 : {[f'{p:.3f}' for p in token_probs]}")
                        print(f"평균 신뢰도: {confidence:.3f}")
                else:
                    confidence = 0.5
                    print("실제 확률 정보 없음, 기본값 사용")
                
                return generated_text, confidence
            
            return generated_text
            
    def batch_extract(self, images):
        results = []
        
        for i, image in enumerate(images):
            print(f"처리중 : {i+1}/{len(images)}")
            try:
                text = self.extract_text(image)
                results.append(text)
            except Exception as e:
                print(f"이미지 {i+1} 처리 실패 : {e}")
                results.append("")
                
        return results
    
    def compare_models(self, image):
        
        models = {
            "Base Printed": "microsoft/trocr-base-printed",
            "Base Handwritten": "microsoft/trocr-base-handwritten",
            "Large Printed": "microsoft/trocr-large-printed",
        }
        
        results = {}
        
        for model_name, model_path in models.items():
            try:
                print(f"모델 테스트 중... : {model_name}")
                
                temp_processor = TrOCRProcessor.from_pretrained(model_path)
                temp_model = VisionEncoderDecoderModel.from_pretrained(model_path)
                temp_model.to(self.device)
                
                #이미지 처리
                pixel_values = temp_processor(image, return_tensors="pt").pixel_values.to(self.device)
                
                #텍스트 생성
                with torch.no_grad():
                    generated_ids = temp_model.generate(pixel_values)
                    
                text = temp_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results[model_name] = text
                
                del temp_processor, temp_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"{model_name} 모델 테스트 실패 : {e}")
                results[model_name] = f"Error : {str(e)}"
                
        return results
    
def trocr_basic_demo():
    
    print("TrOCR 기본 데모 시작")
    ocr = TrOCRSystem()
    
    #1.로컬 이미지 생성 및 테스트
    sample_image = create_sample_trocr_image()
    text = ocr.extract_text(sample_image)
    print(f"인식 결과 : {text}")
    
    #2.다양한 모델 비교
    print("\n2. 다양한 모델 성능 비교")
    try:
        sample_image = create_sample_trocr_image()
        comparison = ocr.compare_models(sample_image)
        
        for model_name, result in comparison.items():
            print(f"\n{model_name} 모델 결과 : {result}")
            
    except Exception as e:
        print(f"모델 비교 중 오류 발생 : {e}")
        
    #3. 신뢰도 계산 테스트
    print("\n3. 신뢰도 계산 테스트")
    try:
        sample_image = create_sample_trocr_image()
        text, confidence = ocr.extract_text(sample_image, return_confidence=True)
        print(f"인식 결과 : {text}")
        print(f"신뢰도 : {confidence:.3f}")
        
    except Exception as e:
        print(f"신뢰도 계산 중 오류 발생 : {e}")
        
    return ocr
    
def create_sample_trocr_image():
    
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new("RGB", (400, 100), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    text = "TrOCR Demo Text"
    draw.text((50, 30), text, font=font, fill="black")
    img.save("sample_trocr_image.png")
    return img

def trocr_performance_test():
    
    print ("\nTrOCR 성능 테스트")
    
    ocr = TrOCRSystem()
    
    test_cases = [
        "Hello, World",
        "The quick brown fox",
        "Machine Learning",
        "2024년 한국어 테스트",
        "Mixed 한글 English 123"
    ]
        
    accuracies = []
    
    for test_text in test_cases:
        print(f"\n테스트 중 : {test_text}")
    
        test_image = create_test_image(test_text)
        
        try:
            predicted = ocr.extract_text(test_image)
        except Exception as e:
            print(f"오류 발생 : {e}")
            predicted = ""        
            
        accuracy = calculate_simple_accuracy(test_text, predicted)
        accuracies.append(accuracy)
        
        print(f"원본 : {test_text}")
        print(f"예측 : {predicted}")
        print(f"정확도 : {accuracy:.2f}")
        
    #전체 평균 정확도 계산
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    print(f"\n전체 평균 정확도 : {avg_accuracy:.2f}")
    
    return avg_accuracy

def create_test_image(text):
    
    from PIL import Image, ImageDraw, ImageFont
    
    img_width = len(text) * 20 + 100
    img_height = 60
    
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        
    draw.text((20, 20), text, font=font, fill="black")
    return img
    
def calculate_simple_accuracy(ground_truth, predicted):
    
    if not ground_truth or not predicted:
        return 0.0
    
    gt_chars = set(ground_truth.lower().replace(" ", ""))
    pred_chars = set(predicted.lower().replace(" ", ""))
    
    if len(gt_chars) == 0:
        return 1.0 if len(pred_chars) == 0 else 0.0
    
    intersection =  len(gt_chars.intersection(pred_chars))
    union = len(gt_chars.union(pred_chars))
    
    return intersection / union if union > 0 else 0.0


try:
    ocr_system = trocr_basic_demo()
    
    performance = trocr_performance_test()
    print(f"\n테스트 완료 - 최종 성능 : {performance:.2f}")
    
except Exception as e:
    print(f"오류 발생 : {e}")




