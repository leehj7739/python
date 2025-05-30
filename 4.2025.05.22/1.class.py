# 다음 클래스를 구현하세요:
# • `Book`: 도서 정보(제목, 저자, ISBN, 출판연도 등)를 관리
# • `Library`: 도서 컬렉션을 관리하고 대출/반납 기능 제공
# • `Member`: 도서관 회원 정보와 대출 목록 관리
# • 다음 기능을 구현하세요:
# • 도서 추가/삭제
# • 도서 검색(제목, 저자, ISBN으로)
# • 도서 대출/반납
# • 회원 등록/관리
# • 회원별 대출 현황 확인
# • 객체 지향 설계 원칙(SOLID)을 최소한 2가지 이상 적용하세요.
# • 적절한 캡슐화를 통해 데이터를 보호하세요.



class Book:
    
    #`Book`: 도서 정보(제목, 저자, ISBN, 출판연도 등)를 관리
    def __init__(self, subject, author, isbn, year):
        self.subject = subject
        self.author = author
        self.isbn = isbn
        self.year = year        


# • `Library`: 도서 컬렉션을 관리하고 대출/반납 기능 제공
class Library:
    
    def __init__(self):
        self.books = []
        self.members = []
        
    #도서 추가
    def add_book(self, book):
        self.books.append(book)
        print(f"도서 추가: {book.subject} , {book.author} , {book.isbn} , {book.year}")
    
    #도서 삭제
    def delete_book(self, book):
        bookinfo = f"도서 삭제: {book.subject} , {book.author} , {book.isbn} , {book.year}"
        self.books.remove(book)
        print(bookinfo)
    
    #도서 검색 (제목, 저자, ISBN으로)
    def search_book(self, search_type, value):
        
        searchbook = None
        
        if len(self.books) == 0:
            print("도서가 존재하지 않습니다.")
            return
        
        for book in self.books:
            if search_type == "subject":
                searchbook = book if book.subject == value else None                
            elif search_type == "author":
                searchbook = book if book.author == value else None                
            elif search_type == "isbn":
                searchbook = book if book.isbn == value else None
            else:
                print("검색 조건이 올바르지 않습니다.")
        
        if searchbook:
            print(f"찾은 도서, 제목: {searchbook.subject}, 저자: {searchbook.author}, ISBN: {searchbook.isbn}, 출판연도: {searchbook.year}")
        else:
            print("해당하는 책이 없습니다.")
    
    #회원 등록
    def add_member(self, member):
        self.members.append(member)
        print(f"회원 등록: {member.id} , {member.name}")

    #회원 삭제
    def del_member(self,member):
        if len(self.members) == 0:
            print("회원이 존재하지 않습니다.")
            return
        
        delmember = f"회원 삭제: {member.id} , {member.name}"
        self.members.remove(member)
        print(delmember)
    
    #회원 목록
    def list_members(self):
        if len(self.members) == 0:
            print("회원이 존재하지 않습니다.")
            return
        
        for member in self.members:
            print(f"회원 목록: {member.id} , {member.name}")
    
    #회원별 대출 현황
    def get_borrowed_books(self, member):
        if len(self.members) == 0:
            print("회원이 존재하지 않습니다.")
            return
        
        if member in self.members:
            print(f"회원 목록: {member.id} , {member.name}")
            for book in member.borrowed_books:
                print(f"대출 도서: {book.subject} , {book.author} , {book.isbn} , {book.year}")
            
    #대출
    def borrow_book(self, member, book):
        if book not in self.books:
            print("대출 가능한 도서가 없습니다.")
            return
        
        member.borrowed_books.append(book)
        self.books.remove(book)
        print(f"대출자 {member.name} 대출 도서: {book.subject} , {book.author} , {book.isbn} , {book.year}")
    
    #반납
    def return_book(self, member, book):
        if book not in member.borrowed_books:
            print("반납 가능한 도서가 없습니다.")
            return
        
        member.borrowed_books.remove(book)
        self.books.append(book)
        print(f"반납자 {member.name} 반납 도서: {book.subject} , {book.author} , {book.isbn} , {book.year}")
    


# • `Member`: 도서관 회원 정보와 대출 목록 관리
class Member:
    
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.borrowed_books = []




library = Library()
b1 = Book("책1", "저자1", "ISBN1", "2025")
b2 = Book("책2", "저자2", "ISBN2", "2026")
b3 = Book("책3", "저자3", "ISBN3", "2027")
library.add_book(b1)
library.add_book(b2)
library.add_book(b3)
library.search_book("subject", "책1")


member1 = Member("1", "홍길동")
member2 = Member("2", "이순신")
member3 = Member("3", "임꺽정")
library.add_member(member1)
library.add_member(member2)
library.add_member(member3)
library.list_members()

library.borrow_book(member1, b1)
library.borrow_book(member1, b2)
library.borrow_book(member1, b3)
library.borrow_book(member2, b1)
library.return_book(member1, b1)
library.borrow_book(member2, b1)
library.get_borrowed_books(member1)
library.get_borrowed_books(member2)


