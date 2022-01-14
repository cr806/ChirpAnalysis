import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import Functions as fnc

image_num = 20
data_filename = 'CK_results_test_multi_ROI_1.csv'
image_align = 'vertical'

data, num_subROIs = fnc.get_data(data_filename)

im_path = data['Image Path'][image_num]
x1 = data['ROI_x1'][image_num]
y1 = data['ROI_y1'][image_num]
x2 = data['ROI_x2'][image_num]
y2 = data['ROI_y2'][image_num]

im = fnc.get_image(im_path, image_align)
im = im.crop((x1, y1, x2, y2))

xdata = list(range(im.size[1]))
subROI_xsize = im.size[0] // num_subROIs

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(im)
colours = ('b', 'g', 'r', 'c', 'm', 'k')

for i in range(num_subROIs):
    amp = data[f'amp_{i}'][image_num]
    assym = data[f'assym_{i}'][image_num]
    res = data[f'res_{i}'][image_num]
    gamma = data[f'gamma_{i}'][image_num]
    off = data[f'off_{i}'][image_num]
    ydata = fnc.fano_array(xdata, amp, assym, res, gamma, off)

    ax1.add_patch(patches.Rectangle((subROI_xsize*i, 0),
                                    subROI_xsize,
                                    im.size[1],
                                    linewidth=2,
                                    edgecolor=colours[i % 6],
                                    facecolor='none'))

    sub_image = im.crop((subROI_xsize*i, 0, subROI_xsize*(i+1), im.size[1]))
    collapsed_im = np.array(sub_image).mean(axis=1)
    ax2.scatter(xdata, collapsed_im,
                color=colours[i % 6],
                s=5,
                alpha=0.5,
                label='raw data')
    ax2.plot(xdata, ydata,
             color=colours[i % 6],
             alpha=0.5,
             linewidth=2,
             label=f'ROI {i}')
    ax2.set_title(data['Image Path'][image_num])
    ax2.legend()
plt.show()
