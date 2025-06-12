# 5개의 공개 API URL에 GET 요청을 보냄
# • 세 가지 방식으로 구현하고 성능을 비교합니다:
# • 순차 처리
# • ThreadPoolExecutor 사용
# • asyncio와 aiohttp 사용
# • API_URLS
# • "https://jsonplaceholder.typicode.com/posts/1",
# • "https://jsonplaceholder.typicode.com/posts/2",
# • "https://jsonplaceholder.typicode.com/posts/3",
# • "https://jsonplaceholder.typicode.com/posts/4",
# • "https://jsonplaceholder.typicode.com/posts/5"

import asyncio
import aiohttp
import time
import requests
import concurrent.futures


#웹사이트 URL
websites = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    "https://jsonplaceholder.typicode.com/posts/4",
    "https://jsonplaceholder.typicode.com/posts/5"
]

# 성능 비교용 fetch_time 변수 선언
fetch_time = {"seq_fetch_time" : 0, "thread_fetch_time" : 0, "async_fetch_time" : 0}

# • 순차 처리
def sequential_requests(urls):
    print("*********순차 처리 시작*********")
    start_time = time.time()
    results = []
    for url in urls:
        url_start_time = time.time()
        response = requests.get(url)
        results.append(response.json())
        url_elapsed = time.time() - url_start_time
        print(f"{url} 요청 완료, 소요시간 : {url_elapsed:.2f}초")
        
    end_time = time.time()
    seq_fetch_time = end_time - start_time
    print(f"*********순차 처리 완료: {seq_fetch_time:.2f}초 소요*********")
    fetch_time["seq_fetch_time"] = seq_fetch_time
    return results

#순차 실행
sequential_results = sequential_requests(websites)

# ThreadPoolExecutor 사용

def threadPool_requests(url):
    
    url_start_time = time.time()
    result = requests.get(url)
    url_elapsed = time.time() - url_start_time
    print(f"{url} 요청 완료, 소요시간 : {url_elapsed:.2f}초")
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    print("*********ThreadPoolExecutor 사용 시작*********")
    start_time = time.time()
    future_to_url = {executor.submit(threadPool_requests, url): url for url in websites}
    
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
        except Exception as e:
            print(f"{url} 요청 중 오류 발생 : {e}")
        else:
            print(f"{url} 요청 완료, 결과 : {result}")
            
    end_time = time.time()
    thread_fetch_time = end_time - start_time
    print(f"*********ThreadPoolExecutor 사용 완료: {thread_fetch_time:.2f}초 소요*********")
    fetch_time["thread_fetch_time"] = thread_fetch_time
#asyncio와 aiohttp 사용


#비동기적으로 웹사이트 내용 가져오기 # 예제 코드
async def fetch(session, url):
    print(f"{url} 요청 시작")
    try:
        start_time = time.time()
        async with session.get(url) as response:
            content = await response.text()
            elapsed = time.time() - start_time
            print(f"{url} 응답 완료, {len(content)} 바이트, 소요시간 : {elapsed:.2f}초")
            return url, len(content), elapsed
    except Exception as e:
        print(f"{url} 오류 발생 : {e}")
        return url, 0, 0

#모든 웹사이트 병렬 요청 # 예제 코드
async def fetch_all_parallel(urls):
    start_time = time.time()
    results = []
    
    async with aiohttp.ClientSession() as session:
        # 모든 URL에 대해 동시에 fetch 작업 생성
        tasks = [fetch(session, url) for url in urls]
        # 모든 작업을 병렬로 실행하고 결과 수집
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"병렬 처리 완료: {end_time - start_time:.2f}초 소요")
    return results

#메인 함수
async def main():

    #병렬 처리
    print("\n *********asyncio와 aiohttp 사용/  병렬 처리 시작 *********")
    start_time = time.time()
    parallel_results = await fetch_all_parallel(websites)
    end_time = time.time()
    async_fetch_time = end_time - start_time
    print(f"*********병렬 처리 완료: {async_fetch_time:.2f}초 소요*********")
    fetch_time["async_fetch_time"] = async_fetch_time

#프로그램 실행 # 예제 코드
if __name__ == "__main__":
    asyncio.run(main())
    
    
# 성능 비교 결과 출력
print("\n 성능 비교 결과 , 빠른순서대로 출력")
for i, (key, value) in enumerate(sorted(fetch_time.items(), key=lambda x: x[1])):
    print(f"{i+1}위 : {key} > {value:.2f}초")



