# • 파일 처리기 구현
# • 다양한 유형의 파일(텍스트, CSV, JSON, 바이너리)을 읽고 쓸 수 있어야 합니다
# • 파일이 존재하지 않거나, 권한이 없거나, 형식이 잘못된 경우 등 다양한 오류 상황을 적절히 처리
# • 사용자 정의 예외 계층 구조를 설계하고 구현
# • 오류 발생 시 로깅을 통해 문제를 기록
# • 모든 파일 작업은 컨텍스트 매니저(`with` 구문)를 사용


import os
import csv
import json
import logging

#로깅 설정
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename = 'app.log',
    encoding='utf-8'   
)

# 사용자 정의 예외 # 이름에 custom 추가
class FileNotFoundCustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class FileFormatCustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# 피드백 반영
# 파일 핸들러 맵핑 사용 , 함수형 프로그래밍

# 텍스트 .txt 파일 읽기
def read_txt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except PermissionError as e:
        logging.error(f"텍스트 파일 읽기 권한 오류: {e}")
    except Exception as e:
        logging.error(f"텍스트 파일 읽기 오류: {e}")

    
# CSV 파일 읽기
def read_csv_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            return list(reader)
    except PermissionError as e:
        logging.error(f"CSV 파일 읽기 권한 오류: {e}")
    except Exception as e:
        logging.error(f"CSV 파일 읽기 오류: {e}")
        
# JSON 파일 읽기
def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except PermissionError as e:
        logging.error(f"JSON 파일 읽기 권한 오류: {e}")
    except Exception as e:
        logging.error(f"JSON 파일 읽기 오류: {e}")
    
# 바이너리 파일 읽기
def read_binary_file(file_path):
    try:
        with open(file_path, "rb") as file:
            return file.read()
    except PermissionError as e:
        logging.error(f"바이너리 파일 읽기 권한 오류: {e}")
    except Exception as e:
        logging.error(f"바이너리 파일 읽기 오류: {e}")


# 파일 핸들러로 함수 맵핑
# 일단 read만 맵핑
FILE_HANDLERS = {
       ".txt": {"read": read_txt_file},
       ".csv": {"read": read_csv_file},
       ".json": {"read": read_json_file},
       ".jpg": {"read": read_binary_file}
   }


# FILE_HANDLERS = {
#        ".txt": {"read": read_txt_file, "write": write_txt_file},
#        ".csv": {"read": read_csv_file, "write": write_csv_file},
#        ".json": {"read": read_json_file, "write": write_json_file},
#        ".jpg": {"read": read_binary_file, "write": write_binary_file}
#    }



def get_filepath(filename):
    return os.path.join(os.getcwd(), "source_files", filename)

def read_file(filename):
    file_path = get_filepath(filename)
    _ , ext = os.path.splitext(filename)

    if not os.path.exists(file_path):
        raise FileNotFoundCustomError(f"파일을 찾을 수 없습니다: {file_path}")

    if ext not in FILE_HANDLERS:
        raise FileFormatCustomError(f"지원하지 않는 파일 형식입니다: {ext}")

    return FILE_HANDLERS[ext]["read"](file_path), ext
   

# # 파일 읽기
# def read_file(file_fullname):
#     #읽을 파일 경로 설정
#     source_file_path = os.path.join(os.getcwd(),  "source_files", file_fullname)

#     #파일 존재 확인
#     if not os.path.exists(source_file_path):
#         logging.error(f"파일을 찾을 수 없습니다: {source_file_path}")
#         raise FileNotFoundError(f"파일을 찾을 수 없습니다: {source_file_path}")


#     #파일 판별  -> 텍스트, csv, json, 바이너리 등
#     file_name, file_extension = os.path.splitext(source_file_path)
#     print(f"파일 이름 : {file_name} , 파일 형식 : {file_extension}")


#     # 텍스트 .txt 파일 읽기
#     def read_txt_file(file_path):
#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 return file.read()
#         except PermissionError as e:
#             logging.error(f"텍스트 파일 읽기 권한 오류: {e}")
#         except Exception as e:
#             logging.error(f"텍스트 파일 읽기 오류: {e}")

        
#     # CSV 파일 읽기
#     def read_csv_file(file_path):
#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 reader = csv.reader(file)
#                 return list(reader)
#         except PermissionError as e:
#             logging.error(f"CSV 파일 읽기 권한 오류: {e}")
#         except Exception as e:
#             logging.error(f"CSV 파일 읽기 오류: {e}")
            
#     # JSON 파일 읽기
#     def read_json_file(file_path):
#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 return json.load(file)
#         except PermissionError as e:
#             logging.error(f"JSON 파일 읽기 권한 오류: {e}")
#         except Exception as e:
#             logging.error(f"JSON 파일 읽기 오류: {e}")
        
#     # 바이너리 파일 읽기
#     def read_binary_file(file_path):
#         try:
#             with open(file_path, "rb") as file:
#                 return file.read()
#         except PermissionError as e:
#             logging.error(f"바이너리 파일 읽기 권한 오류: {e}")
#         except Exception as e:
#             logging.error(f"바이너리 파일 읽기 오류: {e}")


#     #파일 형식 확인
#     try:
#         if file_extension == ".txt":
#             read_file = read_txt_file(source_file_path)
#             print(read_file)
#             return read_file , file_extension
            
#         elif file_extension == ".csv":
#             read_file = read_csv_file(source_file_path)
#             print(read_file)
#             return read_file , file_extension
        
#         elif file_extension == ".json":
#             read_file = read_json_file(source_file_path)
#             print(read_file)
#             return read_file , file_extension
#         #바이너리 파일
#         elif file_extension == ".jpg":
#             read_file = read_binary_file(source_file_path)
#             print(read_file)
#             return read_file , file_extension
#         else:
#             logging.error(f"파일 형식 오류: {file_extension}")
#             raise FileFormatError(f"지원하지 않는 파일 형식입니다: {file_extension}")
        
#     except FileFormatError as e:
#         return None,None
    
#     except Exception as e:
#         logging.error(f"파일 읽기 중 오류 발생: {e}")
#         return False


def copy_file(input_file_name, output_file_name):
    
    # 원본 파일 경로 설정
    source_file_path = os.path.join(os.getcwd(),  "source_files", input_file_name)
    
    # 복사할 파일 경로 설정
    output_file_path = os.path.join(os.getcwd(), "output_files", output_file_name)
    
    # 원본파일 불러오기
    output_file , output_file_extension = read_file(input_file_name)
    
    # 파일 읽기 실패 
    if output_file is None or output_file_extension is None:
        logging.error(f"파일 복사 실패: {input_file_name}")
        return False
    
    # 복사할 파일 쓰기
    try:
        if output_file_extension == ".txt":
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(output_file)
                print(f"파일 복사 성공: {output_file_path}")
                
        elif output_file_extension == ".csv":
            with open(output_file_path, "w", encoding="utf-8", newline='') as file:
                csv.writer(file).writerows(output_file)
                print(f"파일 복사 성공: {output_file_path}")
                
        elif output_file_extension == ".json":
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(output_file, file, ensure_ascii=False, indent=4)
                print(f"파일 복사 성공: {output_file_path}")
                
        elif output_file_extension == ".jpg":
            with open(output_file_path, "wb") as file:
                file.write(output_file)
                print(f"파일 복사 성공: {output_file_path}")
        else:
            logging.error(f"지원하지 않는 파일 형식입니다: {output_file_extension}")
            raise FileFormatCustomError(f"지원하지 않는 파일 형식입니다: {output_file_extension}")
    except FileFormatCustomError as e:
        pass
    except Exception as e:
        logging.error(f"파일 쓰기 중 오류 발생: {e}")
        return False
            
            





#파일 불러오기 테스트
myfile = read_file("source_file.txt")
myfile = read_file("source_file.csv")
myfile = read_file("source_file.json")
myfile = read_file("source_file.jpg")
#myfile = read_file("source_file") # 에러 : 지원하지 않는 파일 형식
#myfile = read_file("unknown_file.txt") # 에러 :파일을 찾을 수 없습니다


#파일 복사사 테스트
copy_file("source_file.txt", "copy_file.txt")
copy_file("source_file.csv", "copy_file.csv")
copy_file("source_file.json", "copy_file.json")
copy_file("source_file.jpg", "copy_file.jpg")
copy_file("source_file", "copy_file") # 에러 : 지원하지 않는 파일 형식
copy_file("unknown_file.txt", "unknown_copy_file.txt") # 에러 :파일을 찾을 수 없습니다
