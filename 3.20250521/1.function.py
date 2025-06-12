# CSV 파일을 읽어 딕셔너리 리스트로 변환하는 함수
# 작성
# • 학생 중 성적이 80점 이상인 학생만 필터링
# • 필터링된 학생들의 평균 나이 계산
# • 모든 함수 호출 시간을 측정하는 데코레이터 적용


import csv
import os
from functools import reduce
import time

students_dict = dict()

#csv 파일 경로 확인
filepath = os.path.join(os.path.dirname(__file__), 'student.csv')

#csv 파일 읽기
with open(filepath, 'r', encoding='utf-8') as file:
    #첫줄 생략략
    next(file)
    reader = csv.reader(file)
    for row in reader:
        students_dict[row[0]] = row[1:]
        
# • 모든 함수 호출 시간을 측정하는 데코레이터 적용
def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 함수 실행 시간 : {end_time - start_time:.10f}초")
        return result
    return wrapper


# • 학생 중 성적이 80점 이상인 학생만 필터링
@time_decorator
def filter_students(students_dict):
    filtered_students = dict(filter(lambda x: int(x[1][2]) >= 80 , students_dict.items()))
    return filtered_students


filtered_students_dict = filter_students(students_dict)
print(f"필터링된 학생들 : {filtered_students_dict}")

# • 필터링된 학생들의 평균 나이 계산
@time_decorator
def calculate_average_age(filtered_students):
    # age_sum = reduce(lambda x , y : x + y , [int(i[2]) for i in filtered_students.values()])
    # average_age = age_sum / len(filtered_students)
    
    # 피드백 내용 반영
    # 리스트 컴프리헨션 사용 및  sum 함수 이용하여 코드 간결화
    total_age = sum(int(info[2]) for info in filtered_students.values())
    average_age = total_age / len(filtered_students)
    
    return average_age




    


print(f"평균 나이 : {calculate_average_age(filtered_students_dict)}")



