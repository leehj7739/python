# 여러 개의 숫자를 입력받아 합계를 계산하는 함수를 작성
# • 사용자가 'q'를 입력하면 입력을 중단하고 지금까지 입력한 숫자의 합을 출력

input_numbers = []

while True:
    input_number = input("숫자를 입력하세요: ")
    if input_number == 'q':
        break
    
    try:
        input_numbers.append(int(input_number))
    except ValueError:
        print("올바른 숫자를 입력하세요!")

print(f"* 입력된 숫자의 총합 : {sum(input_numbers)}")
print(f"* 입력된 숫자 : {input_numbers}")
