import csv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import Functions as fnc

image_num = 0
data_filename = 'CK_results_test_multi_ROI_1.csv'

data = []
with open(data_filename, mode='r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        data.append([r.strip() for r in row])

header = data[0]
data = data[1:]


data_dict = {}
for idx, h in enumerate(header):
    data_dict[h] = []
    for d in data:
        if h != 'Image Path':
            data_dict[h].append(float(d[idx]))
        else:
            data_dict[h].append(d[idx])

xdata_len = int(data_dict['ROI_y2'][image_num] -
                data_dict['ROI_y1'][image_num])
xdata = list(range(xdata_len))

num_subROIs = len([h for h in header if 'amp' in h])
im_path = data_dict['Image Path'][image_num]
image_align = 'vertical'


if im_path[-3:] == 'png':
    im = Image.open(im_path)
elif im_path[-3:] == 'csv':
    im = fnc.convert_CSV_to_Image(im_path)

if image_align == 'horizontal':
    im = im.rotate(90, expand=True)

x1 = data_dict['ROI_x1'][image_num]
y1 = data_dict['ROI_y1'][image_num]
x2 = data_dict['ROI_x2'][image_num]
y2 = data_dict['ROI_y2'][image_num]
im = im.crop((x1, y1, x2, y2))

subROI_xsize = im.size[0] // num_subROIs

ydata = []
for i in range(num_subROIs):
    amp = data_dict[f'amp_{i}'][image_num]
    assym = data_dict[f'assym_{i}'][image_num]
    res = data_dict[f'res_{i}'][image_num]
    gamma = data_dict[f'gamma_{i}'][image_num]
    off = data_dict[f'off_{i}'][image_num]
    y = fnc.fano_array(xdata, amp, assym, res, gamma, off)
    ydata.append(y)

collapsed_xdata = list(range(im.size[1]))
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(im)
rects = []
colours = ('b', 'g', 'r', 'c', 'm', 'k')
for i in range(len(ydata)):
    rects.append(patches.Rectangle((subROI_xsize*i, 0),
                                   subROI_xsize,
                                   im.size[1],
                                   linewidth=2,
                                   edgecolor=colours[i % 6],
                                   facecolor='none'))
    ax1.add_patch(rects[i])

    sub_image = im.crop((subROI_xsize*i, 0, subROI_xsize*(i+1), im.size[1]))
    collapsed_im = np.array(sub_image).mean(axis=1)
    ax2.scatter(collapsed_xdata, collapsed_im,
                color=colours[i % 6],
                s=5,
                alpha=0.5,
                label='raw data')
    ax2.plot(xdata, ydata[i],
             color=colours[i % 6],
             alpha=0.5,
             linewidth=2,
             label=f'ROI {i}')
    ax2.set_title(data_dict['Image Path'][image_num])
    ax2.legend()
plt.show()
