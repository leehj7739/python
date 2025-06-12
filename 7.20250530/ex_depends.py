from fastapi import FastAPI, Depends
import uvicorn
import time

app = FastAPI()

#의존성 함수 정의
def get_current_user():
    return {"user_id" : 1, "username" : "john doe"}

#의존성을 사용하는 엔드포인트
@app.get("/users/me")
def read_current_user(current_user: dict = Depends(get_current_user)):
    return current_user

call_count = 0

#Depends 클래스의 캐싱
def expensive_dependency():
    #비용이 많이 드는 연산을 시뮬레이션 하는 의존성 함수
    global call_count
    call_count += 1
    print(f"의존성 함수 호출됨: {call_count}번째")
    
    time.sleep(0.1)
    return {"data":"expensive_computation_result", "call_count":call_count}
    
#의존성 함수 호출
@app.get("/test1")
def endpoint1(data: dict = Depends(expensive_dependency)):
    #단일 의존성을 사용하는 엔드포인트
    return {"endpoint":"test1", "data":data}

@app.get("/test2")
def endpoint2(
    data1: dict = Depends(expensive_dependency), 
    data2: dict = Depends(expensive_dependency)
):
    #같은 의존성을 두 번 사용하는 엔드포인트
    print(f"data1: {data1}")
    print(f"data2: {data2}")
    print(f"data1 == data2 : {data1 == data2}")
    print(f"data1 is data2 : {data1 is data2}")
    
    return {"endpoint":"test2", "data1":data1, "data2":data2}

#캐싱 비활성화
@app.get("/non-cached-counter")
def non_cached_counter_endpoint(
    data1: dict = Depends(expensive_dependency, use_cache=False), 
    data2: dict = Depends(expensive_dependency, use_cache=False)
):

    return {
        "endpoint":"non-cached-counter", 
        "data1":data1, 
        "data2":data2,
        "same_object":data1 is data2
    }

#매개변수가 있는 의존성 사용
def get_query_token(token:str):
    return token

@app.get("/itmes")
def read_items(token:str = Depends(get_query_token)):
    return {"token":token}

#depends와 타입 힌트
from typing import Optional, Dict, Any

def get_user_info(user_id:int) -> Dict[str, Any]:
    return {"user_id":user_id, "name":"john", "email":"john@example.com"}

def get_settings() -> Dict[str, Any]:
    return {"debug":True, "host":"localhost", "port":8000}

@app.get("/user/{user_id}")
def get_user(
    user_id:int,
    user_info: Dict[str, Any] = Depends(get_user_info),
    settings: Dict[str, Any] = Depends(get_settings)
):
    return {"user": user_info, "settings": settings}
    




#의존성 함수 테스트
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)