import os
import json
from datetime import datetime
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
from openaichatbot import OpenAIChatbot

class AdvancedChatbot(OpenAIChatbot):
    
    def __init__(self, api_key:str, model:str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        self.available_functions = {
            "get_weather": self.get_weather,
            "calculate": self.calculate,
            "get_time": self.get_time
        }
        
    def get_weather(self, location:str):
        weather_data = {
            "서울": "맑음, 15도",
            "부산": "흐림, 20도",
            "대구": "비, 18도",
            "광주": "비, 17도",
            "대전": "비, 16도",
            "울산": "비, 15도",
            "세종": "비, 14도",
        }
        return weather_data.get(location, "해당 지역의의 날씨 정보를 찾을 수 없습니다.")
        
    def calculate(self, expression:str):
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except:
            return "계산할수 없는 식입니다."
        
    def get_time(self):
        #현재 시간 조회
        return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    
    def get_function_response(self, user_message:str)-> str:
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "특정 지역의의 날씨 정보를 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "날씨를 조회할 지역 이름"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "수학 계산을 수행합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "계산할 수학 식"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "현재 시간을 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
        
        #사용자 메시지를 대화 기록에 추가
        self.add_message("user", user_message)
        messages = [self.system_message] + self.conversation_history
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if message.tool_calls:
                
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                
                if function_name in self.available_functions:
                    
                    function_response = self.available_functions[function_name](**arguments)
                    self.add_message("assistant", "")
                    self.conversation_history[-1]["tool_calls"] = message.tool_calls
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "content": function_response,
                        "tool_call_id": tool_call.id
                    })
                    
                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages = [self.system_message] + self.conversation_history,
                    )
                    
                    final_message = final_response.choices[0].message.content
                    self.add_message("assistant", final_message)
                    
                    return final_message
                
                else:
                    assistant_message = message.content
                    self.add_message("assistant", assistant_message)
                    return assistant_message
                
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
        
api_key = os.environ.get("OPENAI_API_KEY")

advanced_bot = AdvancedChatbot(api_key)
advanced_bot.set_system_prompt("당신은 친절한 비서입니다. 함수 호출을 통해 날씨 정보, 계산 결과, 현재 시간을 조회할수 있습니다.")
    
    
#날씨 조회 테스트
response = advanced_bot.get_function_response("서울 날씨 어때?")
print(f"질문: 서울 날씨 어때?\n챗봇: {response}")

#계산 테스트
response = advanced_bot.get_function_response("25 곱하기 4는 얼마야?")
print(f"질문: 25 곱하기 4는 얼마야?\n챗봇: {response}")

#시간 조회 테스트
response = advanced_bot.get_function_response("지금 몇 시야")
print(f"질문: 지금 몇 시야?\n챗봇: {response}")




                