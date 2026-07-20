#!/usr/bin/env python
# coding: utf-8

# # BÀI LÀM LAB 2
# ## Môn Nhập môn Phân tích dữ liệu và Học sâu
# 
# Thống kê, trình bày, trực quan hóa, khảo sát phân phối và tương quan dữ liệu.
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

processed_path = locate_file("processed_dulieuxettuyendaihoc.csv")
print("Thư mục làm việc:", BASE_DIR)
print("Dữ liệu Lab 2:", processed_path)


# # LAB 2 — BÀI THỰC HÀNH TRÌNH BÀY DỮ LIỆU
# 
# Lab 2 sử dụng tệp `processed_dulieuxettuyendaihoc.csv` đã được xử lý từ Lab 1.

# In[2]:


data = pd.read_csv(processed_path)
data['DT'] = data['DT'].astype(int)
print('Kích thước dữ liệu Lab 2:', data.shape)


# ## Phần 1. Thống kê dữ liệu

# ### Câu 1. Sắp xếp `DH1` tăng dần

# In[3]:


sorted_dh1 = data.sort_values('DH1', ascending=True)
print('DH1 đã tăng dần:', bool(sorted_dh1['DH1'].is_monotonic_increasing))
display(sorted_dh1[['STT', 'DH1']].head(20))


# ### Câu 2. Sắp xếp `DH2` tăng dần theo nhóm giới tính

# In[4]:


sorted_dh2_gender = data.sort_values(['GT', 'DH2'], ascending=[True, True])
display(sorted_dh2_gender[['STT', 'GT', 'DH2']].head(30))


# ### Câu 3–5. Pivot-table thống kê `DH1`

# In[5]:


def Q1(x):
    return x.quantile(0.25)

def Q2(x):
    return x.quantile(0.50)

def Q3(x):
    return x.quantile(0.75)

Q1.__name__, Q2.__name__, Q3.__name__ = 'Q1', 'Q2', 'Q3'
agg_functions = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', Q1, Q2, Q3]

def make_pivot(index):
    return pd.pivot_table(
        data,
        values='DH1',
        index=index,
        aggfunc=agg_functions,
        dropna=False
    )

print('Pivot DH1 theo KT:')
pivot_kt = make_pivot(['KT'])
display(pivot_kt)

print('Pivot DH1 theo KT và KV:')
pivot_kt_kv = make_pivot(['KT', 'KV'])
display(pivot_kt_kv)

print('Pivot DH1 theo KT, KV và DT:')
pivot_kt_kv_dt = make_pivot(['KT', 'KV', 'DT'])
display(pivot_kt_kv_dt)


# ## Phần 2. Trình bày dữ liệu

# ### Câu 1. Trình bày biến giới tính `GT`

# In[6]:


gt_table = data['GT'].value_counts().rename('Tan_so').to_frame()
gt_table['Tan_suat'] = gt_table['Tan_so'] / len(data)
display(gt_table)

plt.figure(figsize=(7, 4))
gt_table['Tan_so'].plot(kind='bar')
plt.title('Tần số giới tính')
plt.xlabel('Giới tính')
plt.ylabel('Số lượng')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(6, 6))
gt_table['Tan_suat'].plot(kind='pie', autopct='%1.1f%%', ylabel='')
plt.title('Tần suất giới tính')
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 2. Trình bày `US_TBM1`, `US_TBM2`, `US_TBM3`

# In[7]:


us_cols = ['US_TBM1', 'US_TBM2', 'US_TBM3']
display(data[us_cols].describe().T)
for col in us_cols:
    plt.figure(figsize=(7, 4))
    data[col].plot(kind='hist', bins=10, edgecolor='black')
    plt.title(f'Phân phối {col}')
    plt.xlabel('Điểm thang 4')
    plt.ylabel('Tần số')
    plt.tight_layout()
    plt.show()
    plt.close()


# ### Câu 3. Trình bày biến dân tộc `DT` với học sinh nam

# In[8]:


male_dt = data.loc[data['GT'] == 'M', 'DT']
male_dt_table = male_dt.value_counts().sort_index().rename('Tan_so').to_frame()
male_dt_table['Tan_suat'] = male_dt_table['Tan_so'] / len(male_dt)
display(male_dt_table)

plt.figure(figsize=(7, 4))
male_dt_table['Tan_so'].plot(kind='bar')
plt.title('Dân tộc của học sinh nam')
plt.xlabel('Mã dân tộc')
plt.ylabel('Số lượng')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 4. Trình bày `KV` của học sinh nam, dân tộc Kinh, đủ điều kiện điểm

# In[9]:


condition = (
    (data['GT'] == 'M') &
    (data['DT'] == 0) &
    (data['DH1'] >= 5.0) &
    (data['DH2'] >= 4.0) &
    (data['DH3'] >= 4.0)
)
kv_filtered = data.loc[condition, 'KV']
kv_table = kv_filtered.value_counts().sort_index().rename('Tan_so').to_frame()
if len(kv_filtered):
    kv_table['Tan_suat'] = kv_table['Tan_so'] / len(kv_filtered)
display(kv_table)
print('Số học sinh thỏa điều kiện:', len(kv_filtered))

if not kv_table.empty:
    plt.figure(figsize=(7, 4))
    kv_table['Tan_so'].plot(kind='bar')
    plt.title('Khu vực của nhóm học sinh thỏa điều kiện')
    plt.xlabel('Khu vực')
    plt.ylabel('Số lượng')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()


# ### Câu 5. Trình bày `DH1`, `DH2`, `DH3` từ 5 điểm trở lên ở khu vực `2NT`

# In[10]:


for col in ['DH1', 'DH2', 'DH3']:
    subset = data.loc[(data['KV'] == '2NT') & (data[col] >= 5.0), col]
    print(f'\n{col}: số quan sát = {len(subset)}')
    display(subset.describe().to_frame(name=col).T)
    if not subset.empty:
        plt.figure(figsize=(7, 4))
        subset.plot(kind='hist', bins=8, edgecolor='black')
        plt.title(f'{col} >= 5.0 tại khu vực 2NT')
        plt.xlabel(col)
        plt.ylabel('Tần số')
        plt.tight_layout()
        plt.show()
        plt.close()


# ## Phần 3. Trực quan hóa dữ liệu theo nhóm phân loại

# ### Câu 1. Học sinh nữ trên các nhóm `XL1`, `XL2`, `XL3` dạng unstacked

# In[11]:


female = data[data['GT'] == 'F']
classification_order = ['Y', 'TB', 'K', 'G', 'XS']
xl_female = pd.DataFrame({
    col: female[col].value_counts().reindex(classification_order, fill_value=0)
    for col in ['XL1', 'XL2', 'XL3']
}).T
xl_female.index.name = 'Nam_hoc'
display(xl_female)

plt.figure(figsize=(9, 5))
xl_female.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Xếp loại học sinh nữ theo ba năm')
plt.xlabel('Nhóm xếp loại')
plt.ylabel('Số học sinh')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 2. `KQXT` của khối A, A1, B thuộc khu vực 1, 2

# In[12]:


subset = data[data['KT'].isin(['A', 'A1', 'B']) & data['KV'].isin(['1', '2'])]
kq_kt_kv = pd.crosstab([subset['KT'], subset['KV']], subset['KQXT']).rename(columns={0: 'Rot', 1: 'Dau'})
display(kq_kt_kv)

plt.figure(figsize=(9, 5))
kq_kt_kv.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Kết quả xét tuyển theo khối thi và khu vực')
plt.xlabel('Khối thi, khu vực')
plt.ylabel('Số lượng')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 3. Số lượng thí sinh từng khu vực theo khối thi

# In[13]:


kt_kv_counts = pd.crosstab(data['KT'], data['KV'])
display(kt_kv_counts)

plt.figure(figsize=(10, 5))
kt_kv_counts.plot(kind='bar', stacked=False, ax=plt.gca())
plt.title('Số thí sinh theo khối thi và khu vực')
plt.xlabel('Khối thi')
plt.ylabel('Số lượng')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 4–7. Số lượng đậu/rớt theo khối thi, khu vực, dân tộc và giới tính

# In[14]:


def plot_pass_fail(group_col, title, xlabel):
    table = pd.crosstab(data[group_col], data['KQXT']).rename(columns={0: 'Rot', 1: 'Dau'})
    display(table)
    plt.figure(figsize=(9, 5))
    table.plot(kind='bar', stacked=False, ax=plt.gca())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Số lượng')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()

plot_pass_fail('KT', 'Đậu/rớt theo khối thi', 'Khối thi')
plot_pass_fail('KV', 'Đậu/rớt theo khu vực', 'Khu vực')
plot_pass_fail('DT', 'Đậu/rớt theo dân tộc', 'Mã dân tộc')
plot_pass_fail('GT', 'Đậu/rớt theo giới tính', 'Giới tính')


# ## Phần 4. Trực quan hóa dữ liệu nâng cao

# ### Câu 1. Biểu đồ đường Simple cho biến `T1`

# In[15]:


t1_frequency = data['T1'].value_counts().sort_index()
plt.figure(figsize=(11, 5))
t1_frequency.plot(kind='line', marker='o')
plt.title('Biểu đồ đường tần số của T1')
plt.xlabel('Điểm T1')
plt.ylabel('Tần số')
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 2–3. Tạo biến `phanlopT1` và lập bảng tần số

# In[16]:


data['phanlopT1'] = pd.cut(
    data['T1'],
    bins=[0, 5, 7, 8, np.inf],
    labels=['Kem', 'Trung binh', 'Kha', 'Gioi'],
    right=False,
    include_lowest=True
)
phanlop_t1_table = data['phanlopT1'].value_counts(sort=False).rename('Tan_so').to_frame()
phanlop_t1_table['Tan_suat'] = phanlop_t1_table['Tan_so'] / len(data)
display(phanlop_t1_table)


# ### Câu 4. Biểu đồ Multiple Line cho `T1` theo `phanlopT1`

# In[17]:


t1_group_frequency = pd.crosstab(data['T1'], data['phanlopT1'])
plt.figure(figsize=(11, 5))
t1_group_frequency.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Multiple Line: tần số T1 theo phân lớp')
plt.xlabel('Điểm T1')
plt.ylabel('Tần số')
plt.tight_layout()
plt.show()
plt.close()


# ### Câu 5. Biểu đồ Drop-line cho `T1` theo `phanlopT1`

# In[18]:


plt.figure(figsize=(12, 6))
ax = plt.gca()
for category in data['phanlopT1'].cat.categories:
    series = t1_group_frequency[category]
    mask = series > 0
    if mask.any():
        markerline, stemlines, baseline = ax.stem(
            series.index[mask], series[mask],
            linefmt='-', markerfmt='o', basefmt=' ', label=str(category)
        )
plt.title('Drop-line: tần số T1 theo phân lớp')
plt.xlabel('Điểm T1')
plt.ylabel('Tần số')
plt.legend(title='Phân lớp T1')
plt.tight_layout()
plt.show()
plt.close()


# ## Phần 5. Mô tả dữ liệu và khảo sát dạng phân phối

# ### Câu 1. Mô tả và khảo sát phân phối của `T1`

# In[19]:


t1 = data['T1'].dropna()
q1 = t1.quantile(0.25)
median = t1.quantile(0.50)
q3 = t1.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
lower_whisker = t1[t1 >= lower_fence].min()
upper_whisker = t1[t1 <= upper_fence].max()

summary_t1 = pd.Series({
    'Count': t1.count(),
    'Mean': t1.mean(),
    'Median': t1.median(),
    'Mode': ', '.join(map(str, t1.mode().tolist())),
    'Variance': t1.var(),
    'Std': t1.std(),
    'Min': t1.min(),
    'Q1': q1,
    'Q3': q3,
    'Max': t1.max(),
    'IQR': iqr,
    'Skewness': t1.skew(),
    'Excess kurtosis': t1.kurt(),
})
display(summary_t1.to_frame('Gia_tri'))

boxplot_10 = pd.Series({
    '1. Minimum': t1.min(),
    '2. Lower whisker': lower_whisker,
    '3. Q1': q1,
    '4. Median (Q2)': median,
    '5. Q3': q3,
    '6. Upper whisker': upper_whisker,
    '7. Maximum': t1.max(),
    '8. IQR': iqr,
    '9. Lower fence': lower_fence,
    '10. Upper fence': upper_fence,
})
print('10 đại lượng liên quan Box-Plot:')
display(boxplot_10.to_frame('Gia_tri'))

plt.figure(figsize=(8, 4))
plt.boxplot(t1, vert=False)
plt.title('Box-Plot của T1')
plt.xlabel('Điểm T1')
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(t1, bins=10, edgecolor='black')
plt.title('Histogram của T1')
plt.xlabel('Điểm T1')
plt.ylabel('Tần số')
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(7, 5))
stats.probplot(t1, dist='norm', plot=plt)
plt.title('QQ-Plot kiểm tra chuẩn của T1')
plt.tight_layout()
plt.show()
plt.close()

if abs(t1.skew()) < 0.5:
    skew_comment = 'phân phối gần đối xứng'
elif t1.skew() > 0:
    skew_comment = 'phân phối lệch phải'
else:
    skew_comment = 'phân phối lệch trái'

kurt_comment = 'bẹt hơn phân phối chuẩn' if t1.kurt() < 0 else 'nhọn hơn phân phối chuẩn'
print(f'Nhận xét: mean = {t1.mean():.3f}, median = {t1.median():.3f}, skewness = {t1.skew():.3f}; {skew_comment}.')
print(f'Excess kurtosis = {t1.kurt():.3f}, do đó phân phối {kurt_comment}.')
print('QQ-Plot cho phép đánh giá trực quan; các điểm không hoàn toàn nằm trên đường thẳng nên dữ liệu không hoàn toàn chuẩn.')


# ### Câu 2. Khảo sát phân phối `T1` theo từng nhóm `phanlopT1`

# In[20]:


plt.figure(figsize=(9, 5))
data.boxplot(column='T1', by='phanlopT1', grid=False)
plt.suptitle('')
plt.title('Box-Plot T1 theo phân lớp')
plt.xlabel('Phân lớp T1')
plt.ylabel('Điểm T1')
plt.tight_layout()
plt.show()
plt.close()

for category in data['phanlopT1'].cat.categories:
    values = data.loc[data['phanlopT1'] == category, 'T1'].dropna()
    print(f'\nNhóm {category}: n={len(values)}, mean={values.mean():.3f}, median={values.median():.3f}, skew={values.skew():.3f}')

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=max(4, min(8, values.nunique())), edgecolor='black')
    plt.title(f'Histogram T1 - nhóm {category}')
    plt.xlabel('Điểm T1')
    plt.ylabel('Tần số')
    plt.tight_layout()
    plt.show()
    plt.close()

    if len(values) >= 3:
        plt.figure(figsize=(7, 5))
        stats.probplot(values, dist='norm', plot=plt)
        plt.title(f'QQ-Plot T1 - nhóm {category}')
        plt.tight_layout()
        plt.show()
        plt.close()


# ### Câu 3. Tương quan giữa `DH1` và `T1`

# In[21]:


cov_t1_dh1 = data[['T1', 'DH1']].cov().loc['T1', 'DH1']
corr_t1_dh1 = data[['T1', 'DH1']].corr().loc['T1', 'DH1']
print(f'Covariance(T1, DH1) = {cov_t1_dh1:.4f}')
print(f'Correlation(T1, DH1) = {corr_t1_dh1:.4f}')

plt.figure(figsize=(7, 5))
plt.scatter(data['T1'], data['DH1'], alpha=0.75)
plt.title('Scatter: DH1 theo T1')
plt.xlabel('T1')
plt.ylabel('DH1')
plt.tight_layout()
plt.show()
plt.close()

if abs(corr_t1_dh1) < 0.3:
    print('Nhận xét: mối tương quan tuyến tính giữa T1 và DH1 rất yếu.')
elif abs(corr_t1_dh1) < 0.7:
    print('Nhận xét: mối tương quan tuyến tính giữa T1 và DH1 ở mức vừa.')
else:
    print('Nhận xét: mối tương quan tuyến tính giữa T1 và DH1 mạnh.')


# ### Câu 4. Tương quan giữa `DH1` và `T1` theo từng khu vực

# In[22]:


regional_corr = data.groupby('KV').apply(lambda g: g[['T1', 'DH1']].corr().iloc[0, 1]).rename('Correlation_T1_DH1')
display(regional_corr.to_frame())

for region, group in data.groupby('KV'):
    plt.figure(figsize=(7, 5))
    plt.scatter(group['T1'], group['DH1'], alpha=0.75)
    plt.title(f'DH1 theo T1 - khu vực {region}')
    plt.xlabel('T1')
    plt.ylabel('DH1')
    plt.tight_layout()
    plt.show()
    plt.close()


# ### Câu 5. Tương quan giữa `DH1`, `DH2`, `DH3`

# In[23]:


exam_cols = ['DH1', 'DH2', 'DH3']
print('Ma trận hiệp phương sai:')
display(data[exam_cols].cov())
print('Ma trận tương quan:')
display(data[exam_cols].corr())

pairs = [('DH1', 'DH2'), ('DH1', 'DH3'), ('DH2', 'DH3')]
for x, y in pairs:
    plt.figure(figsize=(7, 5))
    plt.scatter(data[x], data[y], alpha=0.75)
    plt.title(f'Scatter: {y} theo {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()
    plt.close()

print('Nhận xét: các hệ số tương quan có trị tuyệt đối nhỏ, nên ba điểm thi có quan hệ tuyến tính yếu trong mẫu dữ liệu này.')

