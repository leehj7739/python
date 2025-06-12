from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

class OpenAIChatbot:
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        # API 키가 없으면 환경 변수에서 가져오기
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        #대화 기록을 저장하는 리스트 -  각 메시지는 딕셔너리 형태로 저장
        self.conversation_history = []
        #시스템 메시지 - AI의 역할과 행동양식을 정의하는 기본 프롬프트
        self.system_message = {
            "role": "system",
            "content":"당신은 카페 주문을 받는 친절한 직원입니다. 메뉴 추천과 주문처리를 도와주주세요"
        }        

    def add_message(self, role: str, content: str):
        if content is not None:  # None이 아닌 경우에만 추가
            self.conversation_history.append({
                "role": role,
                "content": content
            })
        
    def get_response(self, user_message:str):
        #사용자 메시지에 대한 응답 생성
        #사용자 메시지를 대화 기록에 먼저 추가
        self.add_message("user", user_message)
        
        #시스템 메시지는 대화 히스토리를 결합하여 전체 컨텍스트 구성
        #openai api는 전체 대화 맥락을 필요로함함
        messages = [self.system_message] + self.conversation_history
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                presence_penalty=0.6,
                frequency_penalty=0.0
            )
            
            # 응답이 None이 아닌지 확인
            if response.choices[0].message.content is not None:
                assistant_message = response.choices[0].message.content
                if assistant_message:
                    self.add_message("assistant", assistant_message)
                    return assistant_message
            
            return "죄송합니다. 응답을 생성할 수 없습니다."
            
        except Exception as e:
            error_message = f"오류가 발생했습니다: {str(e)}"
            self.add_message("assistant", error_message)
            return error_message
        
    def clear_history(self):
        #대화 히스토리 초기화
        self.conversation_history = []
        
    def set_system_prompt(self, prompt:str):
        #시스템 프롬프트 설정
        self.system_message["content"] = prompt

if __name__ == "__main__":
            
    api_key = os.environ.get("OPENAI_API_KEY")

    chatbot = OpenAIChatbot(api_key=api_key)

    chatbot.set_system_prompt(
        "당신은 카페 주문을 받는 친절한 직원입니다."
        "메뉴 추천과 주문처리를 도와주주세요"
    )

    user_input = "안녕하세요, 추천메듀가 있나요?"
    response = chatbot.get_response(user_input)
    print(f"사용자: {user_input}\n챗봇: {response}")

    #연속적인 대화 - 이전 맥락이 이어짐
    user_input = "달지 않은 음료를 원해요"
    response = chatbot.get_response(user_input)
    print(f"사용자: {user_input}\n챗봇: {response}")