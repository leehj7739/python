import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import random



class CRNN(nn.Module):
    #CRNN모델 구현 CNN + RNN
    def __init__(self, img_height, img_width, num_chars, num_classes, rnn_hidden = 256):
        #CRNN모델 초기화
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        #인식할 문자 종류수
        self.num_chars = num_chars
        #CTC용 클래스 수 ( 문자 개수 + blank + EOS)
        self.num_classes = num_classes
        
        #CNN 백본
        self.cnn = nn.Sequential(
            
            #Layer1 기본 특징 추출
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            #layer2 더 복잡한 특징 추출
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            #layer3 고수준 특징 추출
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            #layer4 특징 정제
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,1),
            
            #layer5 더 깊은 특징 추출
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            #layer6 특징 강화화
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,1),
            
            #layer7 최종 특징 맵 생성성
            nn.Conv2d(512,512,2,1,0),
            nn.ReLU()
            
        )
        
        #RNN 순차 정보 처리
        #텍스트의 순서 정보를 처리하는 양방향 LSTM
        self.rnn = nn.LSTM(512,rnn_hidden, bidirectional=True, batch_first=True)
        
        #출력 레이어 - RNN 출력을 문자 클래스로 변환
        self.linear = nn.Linear(rnn_hidden * 2 , num_classes)
        # *2 는 양방향 LSTM 이므로
        
    #순전파
    def forward(self,x):
        
        #CNN으로 특징 추출
        conv_features = self.cnn(x)
        
        #RNN 입력을 위해 차원 변경
        b, c, h, w = conv_features.size()
        
        conv_features = conv_features.view(b,c*h,w)
        conv_features = conv_features.permute(0,2,1)
        
        #RNN으로 순차정보 처리
        rnn_out, _ = self.rnn(conv_features)
        
        #출력 레이어
        output = self.linear(rnn_out)
        
        
        #각 문자 위치에 대해 num_classes 클래스중 하나를 예측
        output = F.log_softmax(output,dim=2)
        #로그 확률로 변환 ,마지막 차원에에 소프트 맥스 적용용
        
        return output
    
class SyntheticTextDataset(Dataset):
    #합성 텍스트 데이터 셋 OCR 훈련용 인공 텍스트 이미지 생성
    
    def __init__(self, num_samples=1000, img_height=32, img_width=128, max_text_len = 6):
        #합성 텍스트 데이터셋 초기화
        
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_len = max_text_len
        
        #문자 집합 정의
        self.chars = string.digits
        self.char_to_idx = {char : idx+1 for idx, char in enumerate(self.chars)}
        self.char_to_idx["<blank>"] = 0
        
        self.idx_to_char = {idx : char for char, idx in self.char_to_idx.items()}
        
        self.num_classes = len(self.chars) + 1
        
        #이미지 전처리 변환
        #PIL Image -> tensor / [-1,1]범위로 정규화
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
        
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        #랜덤 텍스트 생성(2~4글자)
        text_len = random.randint(2, min(4,self.max_text_len))
        text = "".join(random.choices(self.chars,k=text_len))
        
        #텍스트로 부터 이미지 생성
        image = self.create_text_image(text)
        
        #텍스트를 인덱스 라벨로 변환환
        label = [self.char_to_idx[char] for char in text]
        
        #텐서 변환
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long), text
    
    def create_text_image(self,text):
        #텍스트로 부터 이미지 생성
        
        img = Image.new("L",(self.img_width,self.img_height), 255)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf",24)
        except:
            font = ImageFont.load_default()
            
        try:
            bbox = draw.textbbox((0,0),text,font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
        except AttributeError:
            text_width, text_height = draw.textsize(text,font=font)
            
        x = (self.img_width - text_width) // 2
        y = (self.img_height - text_height) // 2
        
        draw.text((x,y),text,fill=0,font=font)
        
        img_array = np.array(img)
        #가우시안 노이즈
        noise = np.random.normal(0,5,img_array.shape)
        img_array = np.clip(img_array + noise,0,255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
class CRNNTrainer:
    #CRNN 모델 훈련 클래스
    
    def __init__(self, model, device = "cpu"):
        
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        
    def train_epoch(self,dataloader):
        #한 에포크 훈련 수행
        
        self.model.train()
        
        #총 손실값
        total_loss = 0
        #배치수수
        num_batches = 0
        
        for batch_idx, (images, labels, texts) in enumerate(dataloader):
            #데이터를 디바이스로 이동
            images = images.to(self.device)
            
            targets = [target.to(self.device) for target in labels]
            
            #Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            outputs = outputs.permute(1,0,2)
            
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
            
            target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
            
            targets_1d = torch.cat(targets)
            
            #CTC 로스 계싼
            loss = self.criterion(outputs, targets_1d, input_lengths, target_lengths)
            
            #역전파
            loss.backward()
            
            #그래디언트 클리핑 추가 -> 학습 안정성 향상
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            #가중치 업데이트
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"batch {batch_idx} , loss: {loss.item():.4f}")
                
        self.scheduler.step()
        return total_loss / num_batches
    
    def decode_predictions(self, output, dataset):

        
        # argmax 적용
        pred_indices = torch.argmax(output, dim=2)

        
        decoded_texts = []
        for batch_idx in range(pred_indices.size(0)):
            indices = pred_indices[batch_idx].cpu().numpy()

            decoded_chars = []
            prev_idx = -1
            
            for idx in indices:
                if idx != 0 and idx != prev_idx:
                    try:
                        char = dataset.idx_to_char[idx]
                        decoded_chars.append(char)
                    except KeyError:
                        print(f"   Warning: Invalid index {idx}")
                prev_idx = idx
                
            decoded_text = "".join(decoded_chars)
            decoded_texts.append(decoded_text)
        
        return decoded_texts
            
    def evaluate(self,dataloader, dataset, num_samples=10):
        #모델 평가 수행
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets, gt_texts) in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)
                
                # 모델 출력 확인
                print("\nModel output shape:", outputs.shape)
                print("First output sample:", outputs[0, 0, :5])
                
                predicted_texts = self.decode_predictions(outputs, dataset)
                
                #정확도 계산
                for pred, gt in zip(predicted_texts, gt_texts):
                    print(f"\nGround Truth: {gt}")
                    print(f"Predicted: {pred}")
                    print(f"Match: {'V' if pred == gt else 'X'}")
                    
                    if pred == gt:
                        correct += 1
                    total += 1
                    
                    print(f"GT : {gt} | Pred : {pred} | { 'V' if pred == gt else 'X'}")
        accuracy = correct / total if total > 0 else 0
        print(f"\nAccuracy : {accuracy:.4%} ({correct}/{total})")
        return accuracy
    

def collate_fn(batch):
    
    images, labels, texts = zip(*batch)
    
    images = torch.stack(images)
    
    labels = list(labels)
    
    return images, labels, texts

    
def crnn_practice():
    
    print("=== CRNN OCR 실습 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #데이터셋 생성
    print("\n1. 합성 데이터셋 생성")
    train_dataset = SyntheticTextDataset(num_samples=1000, img_height=32, img_width=128)
    test_dataset = SyntheticTextDataset(num_samples=100, img_height=32, img_width=128)
    
    #데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    print(f"훈련 데이터 : {len(train_dataset)} 개, 테스트 데이터 : {len(test_dataset)} 개")
    print(f"문자 집합 : {train_dataset.chars}")
    
    #셈플 데이터 시각화
    sample_image, sample_label, sample_text = train_dataset[0]
    plt.rc("font",family="Malgun Gothic")
    plt.figure(figsize=(10,3))
    plt.imshow(sample_image.squeeze(), cmap="gray")
    plt.title(f"샘플 이미지: {sample_text}")
    plt.axis("off")
    plt.show()
    
    #모델 생성
    print("\n2. 모델 생성")
    model = CRNN(
        img_height=32,
        img_width=128,
        num_chars=len(train_dataset.chars),
        num_classes=train_dataset.num_classes
    )
    print(f"모델 파라미터 수 : {sum(p.numel() for p in model.parameters()):,}")
    
       
    #훈련 설정
    print("\n3. 모델 훈련")
    trainer = CRNNTrainer(model, device)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss = trainer.train_epoch(train_loader)
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        
        print(f"평균 손실 : {avg_loss:.4f}, 학습률 : {current_lr:.6f}")
        
        #중간 평가
        if (epoch+1) % 2 == 0:
            print(f"\n{epoch+1} 에포크 평가 :")
            trainer.evaluate(test_loader, test_dataset, num_samples=5)
            
    #최종 평가
    print("\n4. 최종 평가")
    final_accuracy = trainer.evaluate(test_loader, test_dataset, num_samples=20)
    
    return model, train_dataset, test_dataset



#실행
model, train_dataset, test_dataset = crnn_practice()







