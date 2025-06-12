import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class IntentClassifier:
    #BERT 기반 의더 분류 모델 클래스
    
    def __init__(self, model_name="klue/bert-base"):
        #의도 분류기 초기화
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def prepare_data(self, texts, labels):
        #훈련/예측을 위한 데이터 전처리
        
        #라벨 인코딩
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        #토큰화
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"            
        )
        print(f"encodings : {encodings}")
        print(f"encoded_labels : {encoded_labels}")
        return encodings, encoded_labels
    
    def train(self, train_texts,train_labels):
        #훈련모델 실행
        #모델 초기화
        num_labels = len(set(train_labels))
        
        self.model = BertForSequenceClassification.from_pretrained(
            "klue/bert-base",
            num_labels=num_labels
        )
        #klue/bert-base 한국어 처리에 최적화된 BERT 모델
             
        #데이터 준비
        train_encodings, train_labels_encoded = self.prepare_data(train_texts, train_labels)
        
        #데이터셋 생성
        class IntentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: (val[idx]).detach().clone() for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
        train_dataset = IntentDataset(train_encodings, train_labels_encoded)
        
        batch_size = 16
        
        epochs = 3
        
        learning_rate = 2e-5
        
        weight_decay = 0.01
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        #옵티마이저 설정
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        #훈련련 루프
        self.model.train()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            total_loss = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                #순전파 
                outputs = self.model(**batch)
                loss = outputs.loss
                
                #역전파
                loss.backward()
                
                #그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                
                #가중치 업데이트
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average Loss : {avg_loss:.4f}")
        print("훈련 완료")
        
    def predict(self, text):
        #입력테스트의 의도 예측
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        print(f"inputs : {inputs}")
        
        with torch.no_grad():
            
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = torch.max(predictions).item()

        intent = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return intent, confidence        
    
# 샘플 데이터
train_texts = [
    "안녕하세요", "반갑습니다", "hello",
    "날씨가 어때요?", "비가 와요?", "맑나요?",
    "주문하고 싶어요", "메뉴 보여줘", "배달 가능해?"
]

train_labels = [
    "greeting", "greeting", "greeting",
    "weather", "weather", "weather",
    "order", "order", "order"
]


classfier = IntentClassifier()
classfier.train(train_texts, train_labels)

intent, confidence = classfier.predict("오늘 날씨 어때?")

print(f"의도 : {intent}, 신뢰도 : {confidence}")