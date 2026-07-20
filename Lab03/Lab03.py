#!/usr/bin/env python
# coding: utf-8

# # BÀI LÀM LAB 3
# ## Môn Nhập môn Phân tích dữ liệu và Học sâu
# 
# Làm sạch và tái cấu trúc dữ liệu nhịp tim của bệnh nhân.
# 
# > Notebook đã được tách riêng và có thể chạy độc lập.

# In[1]:


from pathlib import Path
import csv
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

BASE_DIR = Path.cwd()

def locate_file(filename):
    candidates = [BASE_DIR / filename, Path("/mnt/data") / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Không tìm thấy {filename}. Hãy đặt tệp cùng thư mục với notebook."
    )

DATA_HEART = locate_file("patient_heart_rate.csv")
print("Thư mục làm việc:", BASE_DIR)
print("Dữ liệu Lab 3:", DATA_HEART)


# # LAB 3 — LÀM SẠCH DỮ LIỆU CƠ BẢN
# 
# Tệp `patient_heart_rate.csv` cố ý chứa nhiều lỗi: thiếu header, tên và họ nằm chung cột, đơn vị cân nặng không thống nhất, dòng trống, dòng trùng, ký tự non-ASCII, dữ liệu thiếu và nhiều biến được mã hóa trong tên cột.

# ## Vấn đề 1. Thiếu dòng tiêu đề

# In[2]:


heart_columns = ['Id', 'Name', 'Age', 'Weight', 'm0006', 'm0612', 'm1218', 'f0006', 'f0612', 'f1218']
rows = []
with open(DATA_HEART, encoding='utf-8-sig', newline='') as f:
    for row in csv.reader(f):
        # Chuẩn hóa các dòng thiếu/thừa trường về đúng 10 cột.
        normalized = (row + [''] * len(heart_columns))[:len(heart_columns)]
        rows.append(normalized)

heart = pd.DataFrame(rows, columns=heart_columns)
heart = heart.replace(r'^\s*$', np.nan, regex=True)
print('Dữ liệu sau khi bổ sung header:', heart.shape)
display(heart)


# ## Vấn đề 2. Tách `Name` thành `Firstname` và `Lastname`

# In[3]:


name_parts = heart['Name'].astype('string').str.strip().str.split(n=1, expand=True)
heart['Firstname'] = name_parts[0]
heart['Lastname'] = name_parts[1] if name_parts.shape[1] > 1 else pd.NA
heart = heart.drop(columns='Name')
display(heart[['Id', 'Firstname', 'Lastname']].head(10))


# ## Vấn đề 3. Chuẩn hóa cân nặng về kilogram

# In[4]:


def convert_to_kg(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in ['', '-']:
        return np.nan
    number_text = re.sub(r'[^0-9.]', '', text)
    if not number_text:
        return np.nan
    number = float(number_text)
    return number / 2.2 if 'lb' in text else number

heart['Weight_kg'] = heart['Weight'].map(convert_to_kg)
heart = heart.drop(columns='Weight')
heart['Id'] = pd.to_numeric(heart['Id'], errors='coerce')
heart['Age'] = pd.to_numeric(heart['Age'], errors='coerce')
display(heart[['Id', 'Age', 'Weight_kg', 'Firstname', 'Lastname']])


# ## Vấn đề 4. Xóa các dòng rỗng hoàn toàn

# In[5]:


empty_rows = int(heart.isna().all(axis=1).sum())
print('Số dòng rỗng hoàn toàn:', empty_rows)
heart = heart.dropna(how='all').copy()
print('Kích thước sau khi xóa dòng rỗng:', heart.shape)


# ## Vấn đề 5. Xóa dòng trùng lặp

# In[6]:


duplicate_mask = heart.duplicated(subset=['Firstname', 'Lastname', 'Age', 'Weight_kg'], keep=False)
print('Các dòng thuộc nhóm trùng lặp:')
display(heart.loc[duplicate_mask, ['Id', 'Firstname', 'Lastname', 'Age', 'Weight_kg']])
heart = heart.drop_duplicates(subset=['Firstname', 'Lastname', 'Age', 'Weight_kg'], keep='first').copy()
print('Kích thước sau khi loại trùng:', heart.shape)


# ## Vấn đề 6. Xử lý ký tự non-ASCII

# In[7]:


def normalize_ascii(value):
    if pd.isna(value):
        return value
    return unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')

heart['Firstname'] = heart['Firstname'].map(normalize_ascii)
heart['Lastname'] = heart['Lastname'].map(normalize_ascii)
heart['Firstname'] = heart['Firstname'].fillna('Unknown')
heart['Lastname'] = heart['Lastname'].fillna('Unknown')
display(heart[['Id', 'Firstname', 'Lastname']])


# ## Vấn đề 7. Xử lý thiếu `Age` và `Weight`

# In[8]:


missing_age_weight = pd.DataFrame({
    'So_luong_thieu': heart[['Age', 'Weight_kg']].isna().sum(),
    'Ti_le_thieu': heart[['Age', 'Weight_kg']].isna().mean()
})
display(missing_age_weight)

both_missing = heart['Age'].isna() & heart['Weight_kg'].isna()
print('Số dòng thiếu đồng thời Age và Weight:', int(both_missing.sum()))
display(heart.loc[both_missing, ['Id', 'Firstname', 'Lastname', 'Age', 'Weight_kg']])

# Nếu thiếu cả Age lẫn Weight thì xóa dòng.
heart = heart.loc[~both_missing].copy()

# Nếu chỉ thiếu một biến thì thay bằng mean của biến đó.
age_mean = heart['Age'].mean()
weight_mean = heart['Weight_kg'].mean()
heart['Age'] = heart['Age'].fillna(age_mean)
heart['Weight_kg'] = heart['Weight_kg'].fillna(weight_mean)

print(f'Mean Age dùng để thay thế: {age_mean:.4f}')
print(f'Mean Weight_kg dùng để thay thế: {weight_mean:.4f}')
print('Số thiếu còn lại:', heart[['Age', 'Weight_kg']].isna().sum().to_dict())
display(heart[['Id', 'Firstname', 'Lastname', 'Age', 'Weight_kg']])


# ## Vấn đề 8. Phân rã các cột nhịp tim thành `PulseRate`, `Sex`, `Time`

# In[9]:


heart_rate_cols = ['m0006', 'm0612', 'm1218', 'f0006', 'f0612', 'f1218']
heart_long = heart.melt(
    id_vars=['Id', 'Age', 'Weight_kg', 'Firstname', 'Lastname'],
    value_vars=heart_rate_cols,
    var_name='sex_and_time',
    value_name='PulseRate'
)

heart_long['Sex'] = heart_long['sex_and_time'].str[0].map({'m': 'Male', 'f': 'Female'})
time_code = heart_long['sex_and_time'].str[1:]
heart_long['Time'] = time_code.str[:2] + '-' + time_code.str[2:]
heart_long['PulseRate'] = pd.to_numeric(heart_long['PulseRate'].replace('-', np.nan), errors='coerce')

sex_order = pd.CategoricalDtype(['Male', 'Female'], ordered=True)
time_order = pd.CategoricalDtype(['00-06', '06-12', '12-18'], ordered=True)
heart_long['Sex'] = heart_long['Sex'].astype(sex_order)
heart_long['Time'] = heart_long['Time'].astype(time_order)
heart_long = heart_long.drop(columns='sex_and_time').sort_values(['Id', 'Sex', 'Time']).reset_index(drop=True)

display(heart_long.head(18))
print('Kích thước dữ liệu dạng dài:', heart_long.shape)


# ## Câu 11. Khảo sát và xử lý thiếu `PulseRate` theo thứ tự ưu tiên

# In[10]:


missing_pulse_count = int(heart_long['PulseRate'].isna().sum())
missing_pulse_rate = heart_long['PulseRate'].isna().mean()
print(f'Số PulseRate thiếu: {missing_pulse_count}/{len(heart_long)}')
print(f'Tỉ lệ thiếu PulseRate: {missing_pulse_rate:.2%}')

# Hàm áp dụng ba quy tắc cục bộ đầu tiên và mean của từng người.
def impute_within_person(group):
    group = group.sort_values(['Sex', 'Time']).copy()
    values = group['PulseRate'].astype(float).reset_index(drop=True)

    for i in range(len(values)):
        if pd.notna(values.iloc[i]):
            continue

        fill_value = np.nan
        # 1) Trung bình liền trước và liền sau.
        if i - 1 >= 0 and i + 1 < len(values):
            if pd.notna(values.iloc[i - 1]) and pd.notna(values.iloc[i + 1]):
                fill_value = (values.iloc[i - 1] + values.iloc[i + 1]) / 2

        # 2) Trung bình hai giá liền trước.
        if pd.isna(fill_value) and i - 2 >= 0:
            if pd.notna(values.iloc[i - 1]) and pd.notna(values.iloc[i - 2]):
                fill_value = (values.iloc[i - 1] + values.iloc[i - 2]) / 2

        # 3) Trung bình hai giá liền sau.
        if pd.isna(fill_value) and i + 2 < len(values):
            if pd.notna(values.iloc[i + 1]) and pd.notna(values.iloc[i + 2]):
                fill_value = (values.iloc[i + 1] + values.iloc[i + 2]) / 2

        # 4) Mean các giá trị của người đó.
        if pd.isna(fill_value):
            fill_value = values.mean(skipna=True)

        values.iloc[i] = fill_value

    group['PulseRate'] = values.to_numpy()
    return group

imputed_groups = []
for _, person_group in heart_long.groupby('Id', sort=False, observed=True):
    imputed_groups.append(impute_within_person(person_group))
heart_clean = pd.concat(imputed_groups, ignore_index=True)

# 5) Mean theo giới tính.
sex_means = heart_clean.groupby('Sex', observed=True)['PulseRate'].transform('mean')
heart_clean['PulseRate'] = heart_clean['PulseRate'].fillna(sex_means)

# 6) Mean toàn bộ; nếu vẫn không có thì dùng mức ổn định 75 bpm.
heart_clean['PulseRate'] = heart_clean['PulseRate'].fillna(heart_clean['PulseRate'].mean())
heart_clean['PulseRate'] = heart_clean['PulseRate'].fillna(75.0)

print('Số PulseRate còn thiếu:', int(heart_clean['PulseRate'].isna().sum()))
display(heart_clean.head(18))


# ## Câu 12. Rút gọn, reindex và lưu `patient_heart_rate_clean.csv`

# In[11]:


heart_clean = heart_clean[
    ['Id', 'Firstname', 'Lastname', 'Age', 'Weight_kg', 'PulseRate', 'Sex', 'Time']
].copy()
heart_clean['Id'] = heart_clean['Id'].astype(int)
heart_clean['Age'] = heart_clean['Age'].round(2)
heart_clean['Weight_kg'] = heart_clean['Weight_kg'].round(2)
heart_clean['PulseRate'] = heart_clean['PulseRate'].round(2)
heart_clean['Sex'] = heart_clean['Sex'].astype(str)
heart_clean['Time'] = heart_clean['Time'].astype(str)
heart_clean = heart_clean.sort_values(['Id', 'Sex', 'Time']).reset_index(drop=True)

heart_clean_path = BASE_DIR / 'patient_heart_rate_clean.csv'
heart_clean.to_csv(heart_clean_path, index=False)

print('Đã lưu:', heart_clean_path)
print('Kích thước dữ liệu sạch:', heart_clean.shape)
print('Tổng số dữ liệu thiếu:', int(heart_clean.isna().sum().sum()))
print('Số dòng trùng hoàn toàn:', int(heart_clean.duplicated().sum()))
display(heart_clean.head(24))


# # KẾT LUẬN
# 
# - **Lab 1:** Đã xử lý dữ liệu thiếu, tạo điểm trung bình, xếp loại, quy đổi thang điểm và kết quả xét tuyển.
# - **Lab 2:** Đã thực hiện sắp xếp, pivot-table, bảng tần số/tần suất, các biểu đồ phân nhóm, khảo sát phân phối và tương quan.
# - **Lab 3:** Đã xử lý đầy đủ tám nhóm lỗi dữ liệu, chuyển dữ liệu nhịp tim sang dạng dài, điền dữ liệu thiếu theo thứ tự ưu tiên và lưu tập dữ liệu sạch.
