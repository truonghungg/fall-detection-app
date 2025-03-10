import matplotlib.pyplot as plt
import pandas as pd
data_path ='data/dataset.csv'
data = pd.read_csv(data_path)

activity_counts = data['ActivityLabel'].value_counts()


print(activity_counts)
plt.figure(figsize=(10, 6))

activity_counts.plot(kind='bar', color='skyblue')

# Thêm tiêu đề và nhãn
plt.title('Số lượng mẫu của mỗi hành động')
plt.xlabel('Hành động')
plt.ylabel('Số lượng mẫu')
plt.xticks(rotation=45)  
plt.show()
