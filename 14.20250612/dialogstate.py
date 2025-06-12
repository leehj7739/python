from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import uuid
from openaichatbot import OpenAIChatbot
import os
import dotenv
from EntityExtractor import EntityExtractor

dotenv.load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class DialogState:
    #대화 상태 관리 클래스
    #세션을 고유하게 식별하는 ID - 여러 사용자 구분
    session_id: str
    #대화에서 추출한 개체들 ( 이름, 장소, 날씨 등) - 정보 기억및 재사용을 위해 저장
    entities : Dict[str, Any] = field(default_factory=dict)
    #대화 맥락 스택 - 복잡한 대화 흐름 관리를 위해 사용
    context_stack : List[Dict] = field(default_factory=list)
    #사용자개인정보 -개인화된응답제공을 위해 저장
    user_profile : Dict[str, Any] = field(default_factory=dict)
    #전체 대화 기록 -  이전 대화참조및 멕락유지를 위해 필요
    conversation_history : List[Dict] = field(default_factory=list)
    #마지막 업데이트 시간 - 세션 만료 처리 등에 활용
    last_updated : datetime = field(default_factory=datetime.now)
    
    def add_turn(self, user_input:str, bot_response:str, entities:Dict=None):
        # 대화 턴 추가
        turn = {
            #대화 시점 기록 - 시간순 정렬 및 분석용
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "entities": entities or {}            
        }
        
        self.conversation_history.append(turn)
        
        #새로운 개체 정보가있으면 기존 정보에 추가 -  누적 정보 관리
        if entities:
            self.entities.update(entities)
            
        #상태 업데이트 시간 갱신 - 세션 활성도 추적용
        self.last_updated = datetime.now()
        
    def get_context(self, turn:int = 3) -> List[Dict]:
        #최근 N턴의 대화 컨텍스트 반환
        
        return self.conversation_history[-turn:] if self.conversation_history else []
    
    def clear_context(self):
        #대화 컨텍스트 초기화
        #새로운 주제로 대화를 시작하거나 메모리 정리가 필요할때 사용
        self.conversation_history = []
        self.entities = {}
        self.context_stack = []
            
class ConversationManager:
    #대화 관리자 클래스
    
    def __init__(self):
        #여러 세션을 관리하기 위한 딕셔너리 -  동시 다중 사용자 지원
        self.sessions : Dict[str, DialogState] = {}
        #개체 추출기 초기화 - 사용자 입력에서 중요 정보 추출을 위해 필요
        self.entity_extractor = EntityExtractor()
        
    def create_session(self, user_id : str = None) -> str:
        #새로운 대화 세션 생성
        #기존 사용자는 id 재사용, 신규 사용자는 새 id 발급
        session_id = user_id or str(uuid.uuid4())
        self.sessions[session_id] = DialogState(session_id=session_id)
        #세션 생성시 세션 ID를 키로하여 DialogState 객체를 딕셔너리에 저장

        return session_id
    
    def get_session(self, session_id:str) -> Optional[DialogState]:
        #세션 조회
        return self.sessions.get(session_id)
    
    def process_message(self, session_id:str, user_input:str, chatbot) -> str:
        #메시지 처리 및 컨텍스트 관리
        
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        dialog_state = self.sessions[session_id]
        #개체 추출
        entities = self.entity_extractor.extract_entities(user_input)
        #추출된 개체를 딕셔너리 변환환
        entity_dict = {ent["label"]: ent["text"] for ent in entities}
        #컨텍스트를 고려한 프롬프트 생성 - 이전 대화와 추출된 내용을 포함
        context_prompt = self._build_context_prompt(dialog_state, user_input)
        
        #챗봇 응답 생성
        response = chatbot.get_response(context_prompt)
        
        #대화 상태 업데이트 - 현재 턴의 정보를 세션에 저장
        dialog_state.add_turn(
            user_input=user_input, 
            bot_response=response, 
            entities=entity_dict
        )
        #생성된 응답 반환
        return response
    
    def _build_context_prompt(self, dialog_state:DialogState, current_input:str) -> str:
        #컨텍스트를 포함한 프롬프트 구성
        #dialog_state : 현재 대화 상태 - 이전대화와 저장된 정보 참조용
        #current_input : 현재 사용자 입력 -  새로운 질문/요청 내용
        #반환값 : 컨텍스트가 포함된 완성된 프롬프트 문자열
        
        context_info = []
        
        #이전 대화 컨텍스트 - 최근 3턴의 대화로 맥락 제공
        recent_context = dialog_state.get_context(turn=3)
        if recent_context:
            context_info.append("이전 대화:")
            for turn in recent_context:
                #사용자 - 봇 대화 쌍을 순서대로 추가 - 대화 흐름 이해를 위해
                context_info.append(f"사용자 : {turn['user_input']}")
                context_info.append(f"봇 : {turn['bot_response']}")
                
            #저장된 개체 정보 - 이전에 추출된 주요 정보들    
            if dialog_state.entities:
                context_info.append(f"기억된 정보 : {dialog_state.entities}")
                
            #사용자 프로필 - 개인화된 응답을 위한 사용자 정보
            if dialog_state.user_profile:
                context_info.append(f"사용자 정보 {dialog_state.user_profile}")
                
            #현재 입력  - 응답해야할 새로운 질문  / 요청
            context_info.append(f"현재 질문 : {current_input}")
            
            #모든 컨텍스트 정보를 줄바꿈으로 연결하여 하나의 프롬프트로 구성
            return "\n".join(context_info)
    
    def update_user_profile(self, session_id:str, profile_data:Dict):
        #사용자 프로필 업데이트
        
        if session_id in self.sessions:
            self.sessions[session_id].user_profile.update(profile_data)
        
    def get_conversation_summary(self, session_id:str) -> str:
        #대화 요약 생성
        
        dialog_state = self.sessions.get(session_id)
        if not dialog_state or not dialog_state.conversation_history:
            return "대화 내역이 없습니다."
        
        summary_parts = []
        summary_parts.append(f"세션 ID : {session_id}")
        summary_parts.append(f"대화 턴 수 : {len(dialog_state.conversation_history)}")
        summary_parts.append(f"추출된 개체 : {dialog_state.entities}")
        
        return "\n".join(summary_parts)


if __name__ == "__main__":
    
    #대화 관리자 및 챗봇 초기화
    conv_manager = ConversationManager()

    chatbot = OpenAIChatbot(api_key)
    chatbot.set_system_prompt("당신은 친절한 비서입니다.")

    #새 세션 생성
    session_id = conv_manager.create_session("user123")
    #특정 사용자 ID로 새 대화 세션 시작 - 사용자별 대화 맥락분리

    #대화 진행
    response1 = conv_manager.process_message(session_id, "안녕하세요", chatbot)
    print(f"사용자 : 안녕하세요\n챗봇1 : {response1}")

    response2 = conv_manager.process_message(session_id, "제 이름은 김철수 입니다", chatbot)
    print(f"사용자 : 제 이름은 김철수 입니다\n챗봇2 : {response2}")

    #사용자 프로필 업데이트
    conv_manager.update_user_profile(session_id, {"name": "김철수", "age": 30})

    response3 = conv_manager.process_message(session_id, "아까 제가 뭐라고 했죠?", chatbot)
    print(f"사용자 : 아까 제가 뭐라고 했죠?\n챗봇3 : {response3}")
    #이전 대화 참조 질문으로 컨텍스트 관리 기능 테스트

    response4 = conv_manager.process_message(session_id, "제 나이에 맞는 취미를 추천해 주세요", chatbot)
    print(f"사용자 : 제 나이에 맞는 취미를 추천해 주세요\n챗봇4 : {response4}")

    #대화 요약
    summary = conv_manager.get_conversation_summary(session_id)
    print(f"대화 요약 : {summary}")
                    
    #새로운 유저 세션 생성
    new_session_id = conv_manager.create_session("user456")
    response5 = conv_manager.process_message(new_session_id, "안녕하세요", chatbot)
    print(f"사용자 : 안녕하세요\n챗봇5 : {response5}")

    response6 = conv_manager.process_message(new_session_id, "제 이름은 이영희 입니다", chatbot)
    print(f"사용자 : 제 이름은 이영희 입니다\n챗봇6 : {response6}")



            
            
            
            
                
                
            
                
