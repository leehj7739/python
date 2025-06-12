# 소셜 네트워크에서 사용자 간의 관계와 추천 시스템을
# 구현하는 프로그램을 작성
# • 공통 관심사를 갖는 친구 응답
# • 공통 관심사가 없는 친구 응답

#소셜네트워크 SET 선언 및 데이터 입력
social_network = {
    "Alice": ["음악", "영화", "독서"],
    "Bob": ["스포츠", "여행", "음악"],
    "Charlie": ["프로그래밍", "게임", "영화"],
    "David": ["요리", "여행", "사진"],
    "Eve": ["프로그래밍", "독서", "음악"],
    "Frank": ["스포츠", "게임", "요리"],
    "Grace": ["영화", "여행", "독서"]
}

# 공통 관심사를 갖는 친구 응답
def common_interest():
    print("공통 관심사를 갖는 친구 응답")
    print()
    user = input("당신의 이름을 입력해 주세요요 : ")
    if user not in social_network:
        print(f"{user} 이용자가가 존재하지 않습니다.")
        return
    else:
        print(f"{user} 님과 관심사가 같은 친구의 목록입니다.")
        for friend in social_network:
            if friend != user:
                # 교집합 확인 , set생성
                common_interests = set(social_network[user]) & set(social_network[friend])
                if common_interests:
                    print(f"{friend} 님과 공통 관심사 : {common_interests}")

# • 공통 관심사가 없는 친구 응답
def no_common_interest():
    print("공통 관심사가 없는 친구 응답")
    print()
    user = input("당신의 이름을 입력해 주세요요 : ")
    if user not in social_network:
        print(f"{user} 이용자가가 존재하지 않습니다.")
        return
    else:
        print(f"{user} 님과 공통관심사가 없는 친구의 목록입니다.")
        for friend in social_network:
            if friend != user:
                # 교집합 확인 , set생성
                no_common_interests = set(social_network[user]) & set(social_network[friend])
                if not no_common_interests:
                    print(f"{friend} 님의 관심사 : {social_network[friend]}")

common_interest()
no_common_interest()

