import matplotlib.pyplot as plt
import pandas as pd

file_details = [['CentreOfPeak', 'MedianOfCentres', 'Centre'],
               ['Gaussian', 'MedianOfGaussian', 'Mu'],
               ['Fano', 'MedianOfFano', 'Resonance']
               ]

for file1, file2, column in file_details:
    save_name = f'Libra_20230810-000004-{file1}.png'

    data_filename1 = f'Libra_Data_20230810-000004_BAD-ROI_{file1}_NEWDATA.csv'
    data_filename2 = f'Libra_Data_20230810-000004_BAD-ROI_{file2}_NEWDATA.csv'

    data1 = pd.read_csv(data_filename1)
    data2 = pd.read_csv(data_filename2)

    subROIs = data1['subROI'].unique()

    x_data = range(0, len(data2['Median']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for idx in subROIs:
        y_data = data1[data1['subROI'] == subROIs[idx]][column]
        ax1.plot(x_data, y_data, label=f'SubROI: {idx}')
    ax1.legend()
    ax1.set_ylim([190, 280])
    ax1.set_title('SubROI analysis')

    ax2.plot(x_data,data2['Median'])
    ax2.fill_between(x_data, data2['First Quartile'], data2['Third Quartile'], alpha=0.2)
    ax2.set_ylim([190, 280])
    ax2.set_title('Median analysis')
    fig.suptitle(f'Analysis method: {file1}')
    fig.savefig(save_name)
    # plt.pause(2)