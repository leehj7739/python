# • 로그 파일을 한 줄씩 읽는 제너레이터 함수 작성
# • 특정 패턴(예: 'ERROR', 'WARNING' 등)이 포함된 줄만 필터링하는 제너레이터 작성

import os

#로그파일 읽기

def read_log_file():
    file_path = 'log.txt'
    if os.path.exists(file_path):
        with open("log.txt", "r", encoding="utf-8") as file:
            for line in file:
                yield line.strip()
    else:
        print(f"파일을 찾을 수 없습니다: {file_path}")

#필터링 제너레이터

def filter_logs(loglevel):
    for line in read_log_file():
        loglevel_origin = line.split(" ")[3]
        if loglevel_origin == loglevel:
            yield line.strip()


try:
    log_level ="ERROR"
    filter_gen_error = filter_logs(log_level)
    for _ in range(6):
        print(next(filter_gen_error))
except StopIteration:
    print(f"해당 loglevel : {log_level}에 대한 로그가 없습니다.")

try:
    log_level ="WARNING"
    filter_gen_warning = filter_logs(log_level)
    for _ in range(6):
        print(next(filter_gen_warning))
except StopIteration:
    print(f"해당 loglevel : {log_level}에 대한 로그가 없습니다.")

try:
    log_level ="DEBUG"
    filter_gen_debug = filter_logs(log_level)
    for _ in range(6):
        print(next(filter_gen_debug))
except StopIteration:
    print(f"해당 loglevel : {log_level}에 대한 로그가 없습니다.")
    
try:
    log_level ="INFO"
    filter_gen_info = filter_logs(log_level)
    for _ in range(6):
        print(next(filter_gen_info))
except StopIteration:
    print(f"해당 loglevel : {log_level}에 대한 로그가 없습니다.")   
    
try:
    log_level ="CRITICAL"
    filter_gen_critical = filter_logs(log_level)
    for _ in range(6):
        print(next(filter_gen_critical))
except StopIteration:
    print(f"해당 loglevel : {log_level}에 대한 로그가 없습니다.")