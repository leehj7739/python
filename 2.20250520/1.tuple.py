# 과제
# • 주어진 데이터셋에서 튜플을 활용하여 다음 분석을 수행하세요
# • 연도별 판매량 계산
# • 제품별 평균 가격 계산
# • 최대 판매 지역 찾기
# • 분기별 매출 분석
# 데이터: (연도, 분기, 제품, 가격, 판매량, 지역)
sales_data = [
    (2020, 1, "노트북", 1200, 100, "서울"),
    (2020, 1, "스마트폰", 800, 200, "부산"),
    (2020, 2, "노트북", 1200, 150, "서울"),
    (2020, 2, "스마트폰", 800, 250, "대구"),
    (2020, 3, "노트북", 1300, 120, "인천"),
    (2020, 3, "스마트폰", 850, 300, "서울"),
    (2020, 4, "노트북", 1350, 170, "서울"),
    (2020, 4, "스마트폰", 850, 350, "서울"),
    (2021, 1, "노트북", 1400, 160, "서울"),
    (2021, 1, "스마트폰", 900, 220, "부산"),
    (2021, 2, "노트북", 900, 200, "서울"),
    (2021, 2, "스마트폰", 950, 320, "대구"),
    (2021, 3, "노트북", 1500, 140, "부산"),
    (2021, 3, "스마트폰", 950, 320, "대구"),
    (2021, 4, "노트북", 1500, 140, "부산"),
    (2021, 4, "스마트폰", 950, 370, "서울"),
]


# • 연도별 판매량 계산
def year_sales(data):
    print();
    print("1. 연도별 판매량")
    print();
    year_sales_data = [(year, sales) for year,  *some , sales, region in data]
    year_set = set(year for year, sales in year_sales_data)
    for year in year_set:
        year_sales = [sales for data_year, sales in year_sales_data if data_year == year]
        print(f"{year}년 판매량: {sum(year_sales)}")
        
year_sales(sales_data)        

# • 제품별 평균 가격 계산
def product_avg_price(data):
    print();
    print("2. 제품별 평균 가격")
    print();
    product_data = [(product, price) for year, quarter, product, price, *some in data]
    product_set = set(product for product, price in product_data)
    for product in product_set:
        product_price = [price for data_product, price in product_data if data_product == product]
        print(f"{product} 평균 가격: {sum(product_price) / len(product_price)}")
        
product_avg_price(sales_data)

# • 최대 판매 지역 찾기
def max_sales_region(data):
    print();
    print("3. 최대 판매 지역")
    print();
    region_data = [(region , sales) for *somedata, sales , region in data]
    region_set = set(region for region, sales in region_data)
    max_sales = (0,0)
    for region in region_set:
        region_sales = [sales for data_region, sales in region_data if data_region == region]
        print(f"{region} 판매량: {sum(region_sales)}")
        if sum(region_sales) > max_sales[1]:
            max_sales = (region, sum(region_sales))
            
    print(f"최대 판매 지역: {max_sales[0]}, 누적 판매량: {max_sales[1]}")
        
        
max_sales_region(sales_data)


# • 분기별 매출 분석
def quarter_sales(data):
    print();
    print("4. 분기별 매출 분석")
    print();
    quarter_data = [(quarter, sales *price) for year, quarter, *some, price, sales, region in data]
    quarter_set = set(quarter for quarter, total_price in quarter_data)
    for quarter in quarter_set:
        quarter_sales = [total_price for data_quarter, total_price in quarter_data if data_quarter == quarter]
        print(f"{quarter} 분기 매출: {sum(quarter_sales)}")

quarter_sales(sales_data)

