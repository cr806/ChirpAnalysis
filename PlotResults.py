import matplotlib.pyplot as plt
import pandas as pd

data_filename1 = 'Libra_Data_20230810-000004_ROI1_Fano.csv'
data_filename2 = 'Libra_Data_20230810-000004_ROI1_MedianOfFano.csv'

data1 = pd.read_csv(data_filename1)
data2 = pd.read_csv(data_filename2)

subROIs = data1['subROI'].unique()
print(subROIs[0])

x_data = range(0, len(data2['Median']))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# colours = ('b', 'g', 'r', 'c', 'm', 'k')
for idx in subROIs:
    y_data = data1[data1['subROI'] == subROIs[idx]]['Resonance']
    ax1.plot(y_data, label=f'SubROI: {idx}')
ax1.legend()
ax1.set_title('SubROI analysis')
ax2.plot(x_data,data2['Median'])
ax2.fill_between(x_data, data2['First Quartile'], data2['Third Quartile'], alpha=0.2)
ax2.set_title('Median analysis')
plt.show()