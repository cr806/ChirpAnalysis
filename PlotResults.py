import matplotlib.pyplot as plt
import Functions as func

subROIs_to_plot = [0, 1, 2, 3, 4]
data_filename = 'CK_results_test_multi_ROI_1.csv'
image_align = 'vertical'

data, num_subROIs = func.get_data(data_filename)
num_of_results = len(data['ROI_x1'])

fig, ax = plt.subplots(1, 5, figsize=(12, 6), constrained_layout=True)
colours = ('b', 'g', 'r', 'c', 'm', 'k')

for idx, subROI in enumerate(subROIs_to_plot):
    res = []
    gamma = []
    off = []
    fwhm = []
    r2 = []
    for image_num in range(num_of_results):
        res.append(data[f'res_{subROI}'][image_num])
        gamma.append(data[f'gamma_{subROI}'][image_num])
        off.append(data[f'off_{subROI}'][image_num])
        fwhm.append(data[f'FWHM_{subROI}'][image_num])
        r2.append(data[f'r2_{subROI}'][image_num])

    ax[0].plot(res,
               color=colours[idx % 6],
               label=f'subROI-{subROI}')
    ax[0].set_title('Resonance location')

    ax[1].plot(gamma,
               color=colours[idx % 6],
               label=f'subROI-{subROI}')
    ax[1].set_title('Gamma')

    ax[2].plot(off,
               color=colours[idx % 6],
               label=f'subROI-{subROI}')
    ax[2].set_title('Offset')

    ax[3].plot(fwhm,
               color=colours[idx % 6],
               label=f'subROI-{subROI}')
    ax[3].set_title('FWHM')

    ax[4].plot(r2,
               color=colours[idx % 6],
               label=f'subROI-{subROI}')
    ax[4].set_title('R2')
    ax[4].legend(loc='center right')

    plt.subplots_adjust(bottom=0.1, left=0.1)
    plt.pause(0.5)

plt.show()
