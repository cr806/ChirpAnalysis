import matplotlib.pyplot as plt
import pandas as pd

file_details = [['CentreOfPeak', 'MedianOfCentres', 'Centre'],
               ['Gaussian', 'MedianOfGaussian', 'Mu'],
               ['Fano', 'MedianOfFano', 'Resonance']
               ]
# file_details = [['Fano', 'MedianOfFano', 'Resonance']]
ROI_pairs = [[0, 1], [2, 3], [4, 5]]

for file1, file2, column in file_details:
    save_name = f'Libra_20230810-000004-{file1}.png'

    data_filename1 = f'Libra_Data_20230810-000004_CS_{file1}_NEWDATA.csv'
    data_filename2 = f'Libra_Data_20230810-000004_CS_{file2}_NEWDATA.csv'

    data1 = pd.read_csv(data_filename1)
    data2 = pd.read_csv(data_filename2)

    subROIs = data1['subROI'].unique()
    ROISize = data1['ROI_y2'][0] - data1['ROI_y1'][0]

    x_data = range(0, len(data2['Median']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for r1, r2 in ROI_pairs:
        for idx in subROIs:
            filteredData = data1[data1['subROI'] == subROIs[idx]]
            ROI0 = filteredData[filteredData['ROI'] == r1]
            ROI1 = filteredData[filteredData['ROI'] == r2]
            y_data = ROI1[column].to_numpy()
            y_data = ROISize - y_data
            y_data = y_data + ROI0[column].to_numpy()
            ax1.plot(y_data, label=f'SubROI: {idx}')
        # ax1.legend()
        ax1.set_ylim([350, 480])
        ax1.set_title('SubROI analysis')

        ROI0 = data2[data2['ROI'] == r1]
        ROI1 = data2[data2['ROI'] == r2]
        y_data = ROI1['Median'].to_numpy()
        y_data = ROISize - y_data
        y_data = y_data + ROI0['Median'].to_numpy()
        ax2.plot(y_data)
        F = ROI0['First Quartile'].to_numpy() + ROI1['First Quartile'].to_numpy()
        T = ROI0['Third Quartile'].to_numpy() + ROI1['Third Quartile'].to_numpy()
        IQR = T - F
        ax2.fill_between(range(len(y_data)), y_data-IQR/2, y_data+IQR/2, alpha=0.2)
        ax2.set_ylim([350, 480])
        ax2.set_title('Median analysis')
    fig.suptitle(f'Analysis method: Bowtie - {file1}')
    # fig.savefig(save_name)
    plt.pause(2)