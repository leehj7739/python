# 학생 리스트 [[이름, 점수], ... ]
studentInfo_list = []
# 초기 학생 정보 추가
studentInfo_list.append(("alice", 100))
studentInfo_list.append(("bob", 90))
studentInfo_list.append(("carol", 80))
studentInfo_list.append(("홍길동", 70))
studentInfo_list.append(("강감찬", 60))    

# 학생 추가
def add_student(name, score):
    try:
        score = int(score)
        studentInfo_list.append((name, score))
        print(f"학생 정보가 추가되었습니다. 이름 : {name}, 점수: {score}")
    except:
        print("올바른 숫자를 입력해주세요.")
    return


# 학생 삭제
def remove_student(name):
    if len(studentInfo_list) == 0:
        print("등록된 학생이 없습니다.")
        return
    
    for student in studentInfo_list:
        if (student[0] == name):
            studentInfo_list.remove(student)
            print(f"학생 정보가 삭제되었습니다. 이름 : {name}")
            return
        
    print(f"학생 정보가 존재하지 않습니다. 이름 : {name}")

# 성적 수정  튜플 <-> 리스트
def modify_score(name):
    temp_list = list(studentInfo_list)
    
    for i in range(len(temp_list)):
        if temp_list[i][0] == name: 
            print(f"이름 : {name}, 현재 점수 : {temp_list[i][1]}")
            try:
                input_score = int(input("수정할 점수를 입력하세요: "))
                temp_list[i] = list(temp_list[i])
                temp_list[i][1] = input_score 
                #다시 스튜던트 리스트로로
                studentInfo_list = tuple(temp_list)
                print(f"학생 정보가 수정되었습니다. 이름 : {name}, 점수: {studentInfo_list[i][1]}")
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
            return 
        
    print(f"학생 정보가 존재하지 않습니다. 이름 : {name}")       
     
    

# 전체 목록 출력
def show_all_student_info():
    if len(studentInfo_list) == 0:
        print("등록된 학생이 없습니다.")
        return
    
    print("= 학생 정보 =")
    for i, (name, score) in enumerate(studentInfo_list , 1):
        print(f"{i}번째 학생 정보 - 이름 : {name}, 점수: {score}")
        

def show_student_stats():
    if len(studentInfo_list) == 0:
        print("등록된 학생이 없습니다.")
        return
    
    # 첫번째 학생 점수로 초기화
    total_score = 0
    average_score = studentInfo_list[0][1]
    max_score = studentInfo_list[0][1]
    min_score = studentInfo_list[0][1]
    
    for student in studentInfo_list:
        total_score += student[1]
        print(total_score)
        if student[1] > max_score:
            max_score = student[1]
        if student[1] < min_score:
            min_score = student[1]
        
    average_score = total_score / len(studentInfo_list)
    
    print("= 학생 통계 =")
    print(f"학생 숫자 : {len(studentInfo_list)}")
    print(f"평균 : {average_score}")
    print(f"최고점 : {max_score}")
    print(f"최저점 : {min_score}")
    

# 프로그램 시작
while True:
    print("###########################")
    print("원하시는 작업 번호를 입력하세요")
    print("1. 학생 추가")
    print("2. 학생 삭제")
    print("3. 성적 수정")
    print("4. 전체 목록 출력")
    print("5. 학생 통계 출력")
    print("6. 프로그램 종료")
    choice = input("작업 번호를 입력하세요: ")
    if choice == "1":
        name = input("추가할 학생의 이름을 입력하세요: ")
        score = input("추가할 학생의 점수를 입력하세요: ")
        add_student(name, score)
    elif choice == "2":
        name = input("삭제할 학생의 이름을 입력하세요: ")
        remove_student(name)
    elif choice == "3":
        name = input("수정할 학생의 이름을 입력하세요: ")
        modify_score(name)
    elif choice == "4":
        show_all_student_info()
    elif choice == "5":
        show_student_stats()
    elif choice == "6": 
        print("프로그램을 종료합니다.")
        break
    else:
        print("잘못된 작업 번호입니다. 다시 입력해주세요.")
