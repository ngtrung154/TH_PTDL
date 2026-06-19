# Phần 1: Khởi tạo và Đọc dữ liệu

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

df = load_data('ecommerce_sales_data.xlsx')
print(df.head(7))
print(df.tail(5))
print(df.info())
print(df['Quantity'].agg(['min', 'max', 'mean']))
print(df['UnitPrice'].agg(['min', 'max', 'mean']))
print(df['Discount'].agg(['min', 'max', 'mean']))

# Phần 2: Khám phá và Làm sạch dữ liệu

print(df.isnull().sum())
df['Discount'].fillna(0, inplace=True)
df['CustomerID'].fillna('GUEST', inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
print(df.info())
if df.duplicated().sum() > 0:
    print("Có dữ liệu trùng lặp.")
    df.drop_duplicates()

# Phần 3: Trích xuất và Biến đổi dữ liệu (Feature Engineering)
df['Revenue'] = df['Quantity'] * df['UnitPrice'] * (1 - df['Discount'])
df['Year']   = df['Date'].dt.strftime('%Y').astype(int)
df['Month'] = df['Date'].dt.strftime('%m').astype(int)
df['DateOfWeek']   = df['Date'].dt.strftime('%A').astype(str)
df['Price_Segment'] = pd.cut(df['UnitPrice'], bins=[0, 50, 150, float('inf')], labels=['Low', 'Medium', 'High'])
print(df.head(10))

# Phần 4: Lọc và Truy vấn dữ liệu
print(df.query("Region == 'North' and Revenue > 300"))
print(df.query("Product_Category == 'Electronics' and Discount == 0"))
print(df.query("Month >=  3 and Month <= 6"))
print(df[[ 'OrderID' , 'Product_Name' , 'Revenue' ]].sort_values(by='Revenue', ascending=False).head(10))
print(df.loc[df["Quantity"].idxmax()])

# Phần 5: Phân tích, Gom nhóm và Tổng hợp
print(df.groupby('Region')['Revenue'].sum())
print(df.groupby('Product_Category')[['Quantity', 'UnitPrice']].mean())
print(df.groupby('Month')['OrderID'].count())
print(df[df['CustomerID'] != 'GUEST'].groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False).head(3))

# Phần 6: Trực quan hóa dữ liệu cơ bản
sns.barplot(x='Product_Category', y='Revenue', data=df)
plt.title('Tong doanh thu theo danh muc san pham')
plt.xlabel('Product_Category')
plt.ylabel('Revenue')
plt.show()

sns.lineplot(x='Month', y='Revenue', data=df)
plt.title('Tong doanh thu theo thang trong nam')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()