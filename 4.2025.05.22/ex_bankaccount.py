class BankAccount:
    
    #클래스 변수
    #이율
    interest_rate = 0.01

    def __init__(self, owner, balance = 0):
        self.owner = owner
        self.balance = balance
        self.transaction_history = []
        self._log_transaction("계좌 개설", balance)
        
    #입금
    def deposit(self, amount):
        if amount <= 0:
            print("입금액은 0보다 커야 합니다")
            return False
        
        self.balance += amount
        self._log_transaction("입금", amount)
        print(f"{amount}원이 입금되었습니다. 현재 잔액: {self.balance:,}원")
        return True

    #출금
    def withdraw(self, amount):
        if amount <= 0:
            print("출금액은 0보다 커야 합니다")
            return False

        if amount > self.balance:
            print("잔액부족. 현재 잔액 {self.balance:,}원")
            return False
        
        self.balance -= amount
        self._log_transaction("출금", amount)
        print(f"{amount}원이 출금되었습니다. 현재 잔액: {self.balance:,}원")
        return True

    #잔액 조회
    def get_balance(self):
        print(f"현재 잔액 : {self.balance:,}원")
        return self.balance
    
    #이자 적용
    def apply_interest(self):
        interest = self.balance * BankAccount.interest_rate
        self.balance += interest
        self._log_transaction("이자", interest)
        print(f"이자 {interest:,.2f}원이 추가되었습니다. 현재 잔액: {self.balance:,}원")

    #거래내역 로깅
    def _log_transaction(self, transaction_type, amount):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.transaction_history.append({
            "type" : transaction_type,
            "amount" : amount,
            "timestamp" : timestamp,
            "balance" : self.balance
        })
    
    #거래내역 출력
    def print_transaction_history(self):
        print(f"\n{self.owner}님의 거래내역 : ")
        print("-" * 60)
        print(f"{'일시':<20}{'종류':<10}{'금액':<15}{'잔액':<15}")
        print("-" * 60)

        for transaction in self.transaction_history:
            print(f"{transaction['timestamp']:<20}"
                  f"{transaction['type']:<10}"
                  f"{transaction['amount']:,}원".ljust(15) + "\t" +
                  f"{transaction['balance']:,}원".ljust(15))
        print("-" * 60)


#계좌 생성
my_account = BankAccount("홍길동", 10000000)
your_account = BankAccount("김철수")

#계좌 조작
my_account.get_balance()
my_account.deposit(500000)
my_account.withdraw(200000)
my_account.withdraw(10000000000000)

#이자적용
my_account.apply_interest()

#거래내역 출력
my_account.print_transaction_history()
