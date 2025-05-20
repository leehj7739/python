# 딕셔너리를 활용하여 간단한 주소록 프로그램 작성
# • 연락처 이름을 키로 하고, 전화번호, 이메일, 주소 등의 정보를 값으로 저장
# • 중첩 딕셔너리 구조를 사용하여 각 연락처마다 여러 정보를 저장
# • 연락처 추가, 삭제, 검색, 수정, 모든 연락처 보기 기능을 구현

# address_book = {
#     "name" : {
#         "phone_number" : "010-1234-5678",
#         "email" : "hong@example.com",
#         "address" : "서울시 강남구"
#     }
# }
#주소록 생성 및 더미 데이터 추가
address_book = {
    "홍길동": {
        "phone_number": "010-1234-5678",
        "email": "hong@example.com",
        "address": "서울시 강남구"
    },
    "김철수": {
        "phone_number": "010-2345-6789",
        "email": "kimcs@example.com",
        "address": "부산시 해운대구"
    },
    "이영희": {
        "phone_number": "010-3456-7890",
        "email": "leeyh@example.com",
        "address": "대구시 수성구"
    },
    "박민수": {
        "phone_number": "010-4567-8901",
        "email": "parkms@example.com",
        "address": "인천시 연수구"
    },
    "최지우": {
        "phone_number": "010-5678-9012",
        "email": "choijw@example.com",
        "address": "광주시 북구"
    }
}



#연락처 추가
def add_contact():
    print("연락처 추가")
    input_name = input("이름을 입력해주세요 : ")
    if input_name  in address_book:
        print(f"{input_name} 이미 존재하는 연락처입니다.")
        return
    else:
        input_phone_number = input("전화번호를 입력해주세요 : ")
        input_email = input("이메일을 입력해주세요 : ")
        input_address = input("주소를 입력해주세요 : ")
        address_book[input_name] = {
            "phone_number" : input_phone_number,
            "email" : input_email,
            "address" : input_address
        }
        print(f"{input_name} 연락처가 추가되었습니다.")
        print(f"전화번호 : {address_book[input_name]["phone_number"]}")
        print(f"이메일 : {address_book[input_name]["email"]}")
        print(f"주소 : {address_book[input_name]["address"]}")
        
#연락처 삭제
def delete_contact():
    print("연락처 삭제")
    input_name = input("삭제할 연락처의 이름을 입력해주세요 : ")
    if input_name not in address_book:
        print(f"{input_name} 연락처가 존재하지 않습니다.")
        return
    else:
        del address_book[input_name]
        print(f"{input_name} 연락처가 삭제되었습니다.")


#연락처 검색
def search_contact():
    print("연락처 검색")
    input_name = input("검색할 연락처의 이름을 입력해주세요 : ")
    if input_name not in address_book:
        print(f"{input_name} 연락처가 존재하지 않습니다.")
        return
    else:
        print(f"{input_name} 연락처 정보")
        print(f"전화번호 : {address_book[input_name]["phone_number"]}")
        print(f"이메일 : {address_book[input_name]["email"]}")
        print(f"주소 : {address_book[input_name]["address"]}")


#연락처 수정
def modify_contact():
    print("연락처 수정")
    input_name = input("수정할 연락처의 이름을 입력해주세요 : ")
    if input_name not in address_book:
        print(f"{input_name} 연락처가 존재하지 않습니다.")
        return
    else:
        print("수정할 내용을 입력하세요")
        input_phone_number = input("전화번호 : ")
        input_email = input("이메일 : ")
        input_address = input("주소 : ")
        new_info = { input_name : 
            {
            "phone_number" : input_phone_number,
            "email" : input_email,
            "address" : input_address
            }
                    }
        address_book.update(new_info)
        print(f"{input_name} 연락처 정보가 수정되었습니다.")
        print(f"{input_name} 연락처 정보")
        print(f"전화번호 : {address_book[input_name]["phone_number"]}")
        print(f"이메일 : {address_book[input_name]["email"]}")
        print(f"주소 : {address_book[input_name]["address"]}")
        
#모든 연락처 보기    
def show_all_contact():
    print(f"모든 연락처 보기, 총 이용자 수 {len(address_book)}명")
    count = 1
    for name, info in address_book.items():
        print(f"{count}번 이용자")
        print(f"이름 : {name}")
        print(f"전화번호 : {info['phone_number']}")
        print(f"이메일 : {info['email']}")
        print(f"주소 : {info['address']}")
        count += 1
    
    
 
while True:
    print("1. 연락처 추가")
    print("2. 연락처 삭제")
    print("3. 연락처 검색")
    print("4. 연락처 수정")
    print("5. 모든 연락처 보기")
    print("6. 종료")
    choice = input("원하는 작업을 선택해주세요 : ")
    if choice == "1":
        add_contact()
    elif choice == "2":
        delete_contact()
    elif choice == "3":
        search_contact()
    elif choice == "4":
        modify_contact()
    elif choice == "5":
        show_all_contact()
    elif choice == "6":
        break
        






