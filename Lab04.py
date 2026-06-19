import pandas as pd
from pyparsing import line
import seaborn as sns
import matplotlib.pyplot as plt

#1
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

df = load_data('Lab04\\titanic_disaster.csv')
print(df.head(10))

#2
missing_data = pd.DataFrame({
    'So_luong_thieu': df.isnull().sum(),
})

missing_data = missing_data[missing_data['So_luong_thieu'] > 0]

print(missing_data)

plt.figure(figsize=(10, 6))
sns.heatmap(missing_data, cbar=True, annot=True, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

#3
idx = df.columns.get_loc('Name')

split_name = df['Name'].str.split(',', expand=True)

df.insert(idx + 1, 'firstName', split_name[0].str.strip())
df.insert(idx + 2, 'secondName', split_name[1].str.strip())

df.drop(columns='Name', inplace=True)

print(df)

#4
df['Sex'].replace({'male': 'M', 'female': 'F'}, inplace=True)
print(df)

#5
sns.boxplot(x='Pclass', y='Age', data=df)

plt.title('Phan bo tuoi theo hang hanh khach')
plt.xlabel('Pclass')
plt.ylabel('Age')
plt.show()

df.groupby('Pclass')['Age'].mean()
print(df['Age'].isnull().sum())
df['Age'] = df.groupby('Pclass')['Age'].transform(
    lambda x: x.fillna(x.mean())
)

print(df[['Pclass', 'Age']])

plt.figure(figsize=(10,5))

sns.heatmap(
    df[['Age']].isnull(),
    cbar=True, annot=True, cmap='viridis'
)

plt.title('Kiem tra du lieu thieu cua cot Age sau khi xu ly')
plt.show()


#6
df['Agegroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, float('inf')], labels=['Kid', 'Teen', 'Adult', 'Older'])
print(df)

#7
df['namePrefix'] = df['secondName'].str.extract(r'([A-Za-z]+)\.')
df['secondName'] = df['secondName'].str.replace(r'([A-Za-z]+)\.', '', regex=True).str.strip()
df.insert(idx + 2, 'namePrefix', df.pop('namePrefix'))

print(df)

#8
family_size = 1 + df['SibSp'] + df['Parch']

idx = df.columns.get_loc('Parch')
df.insert(idx + 1, 'FamilySize', family_size)

print(df)

#9
df['Alone'] = 0
df.loc[df['FamilySize'] == 1, 'Alone'] = 1
print(df)

#10
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['typeCabin'] = df['Cabin'].str[0]
print(df)

#11(BỎ)

#12
print("12. Nhận xét: Phụ nữ sống sót nhiều hơn nam giới.")
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Ty le song sot theo gioi tinh')
plt.show()

#13
print("13. Nhận xét: Hành khách hạng 1 sống sót nhiều nhất, hạng 3 ít nhất.")
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Ty le song sot theo hang hanh khach')
plt.show()

#14
print("14. Nhận xét: Trẻ em và phụ nữ có nhiều người sống sót hơn.")
sns.violinplot(
    data=df,
    x='Sex',
    y='Age',
    hue='Survived',
    split=True
)

plt.title('Phan bo tuoi theo gioi tinh va tinh trang song sot')
plt.show()

#15
print("15. Nhận xét: Người đi cùng gia đình nhỏ thường sống sót nhiều hơn.")
sns.barplot(
    data=df,
    x='FamilySize',
    y='Survived'
)

plt.title('Xac suat song sot theo quy mo gia dinh')
plt.ylabel('Ti le song sot')
plt.show()

#16
print("16. Nhận xét: Người mua vé đắt thường sống sót nhiều hơn.")
sns.scatterplot(
    data=df,
    x='Fare',
    y='Survived'
)

plt.title('Xac suat song sot theo gia ve')
plt.show()

#17
print("17.Nhận xét: Hành khách hạng 1 có nhiều người sống sót hơn ở các cảng lên tàu. Cảng lên tàu cũng có ảnh hưởng đến số người sống sót.")
sns.countplot(
    data=df,
    x='Pclass',
    hue='Survived'
)

plt.title('Song sot theo hang hanh khach')
plt.show()
