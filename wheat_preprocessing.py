import datetime

import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from argparse import ArgumentParser



from matplotlib import pyplot as plt
import scipy.io as sio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = ArgumentParser()
parser.add_argument("--threshold", default=0.2, help="FFT threshold")
parser.add_argument("--exclude", default=120, help="FFT high-pass threshold")

args = parser.parse_args()
args = vars(args)

F_THR = np.double(args['threshold'])

EXCLUDE = int(args['exclude'])


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


class PassDataframe():
    def __init__(self, df):
        self.train_df = df


# @title
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        # print("Img dir: ", self.image_dir)
        # print("Image id: ", image_id)

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def apply_fft(img, use_fft_np, use_complex_cv):
    # Use the np version of fft
    if use_fft_np:
        # --- fft with the np package ---
        img_fft = np.fft.fft2(img)
        # im_fft = np.fft.fftshift(f)
        # magnitude_spectrum_np = 20 * np.log(np.abs(im_fft))

    # Use the opencv version of fft
    else:
        # --- constructing the equivalent img freq with the openCV fft ---
        f_cv = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # f_cv = np.fft.fftshift(f_cv)
        # magnitude_spectrum_cv[:, :, j] = 20 * np.log(cv2.magnitude(f_cv[:, :, 0], f_cv[:, :, 1]))

        if use_complex_cv:
            # Create the complex nr from the 2 dimensions inside the fft decomposition
            img_fft = f_cv[:, :, 0] + 1j * f_cv[:, :, 1]
        else:
            img_fft = cv2.magnitude(f_cv[:, :, 0], f_cv[:, :, 1])

    return np.abs(img_fft) ** 2


def collate_fn(batch):
    return tuple(zip(*batch))


def apply_fft(img, use_fft_np, use_complex_cv):
    # Use the np version of fft
    if use_fft_np:
        # --- fft with the np package ---
        img_fft = np.fft.fft2(img)
        # im_fft = np.fft.fftshift(f)
        # magnitude_spectrum_np = 20 * np.log(np.abs(im_fft))

    # Use the opencv version of fft
    else:
        # --- constructing the equivalent img freq with the openCV fft ---
        f_cv = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # f_cv = np.fft.fftshift(f_cv)
        # magnitude_spectrum_cv[:, :, j] = 20 * np.log(cv2.magnitude(f_cv[:, :, 0], f_cv[:, :, 1]))

        if use_complex_cv:
            # Create the complex nr from the 2 dimensions inside the fft decomposition
            img_fft = f_cv[:, :, 0] + 1j * f_cv[:, :, 1]
        else:
            img_fft = cv2.magnitude(f_cv[:, :, 0], f_cv[:, :, 1])

    return np.abs(img_fft) ** 2


if __name__ == "__main__":

    startTime = datetime.datetime.now()
    print("Startime: ", startTime)

    use_fft_np = False  # (choose between np and opencv implementations for fft)
    use_complex_cv = True

    if use_fft_np:
        suffix = 'np'
    else:
        suffix = 'cv'

    DIR_INPUT = 'wheat'
    DIR_TRAIN = f'{DIR_INPUT}/sampled_images'

    # train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')  # Set index col to be 0
    train_df = pd.read_csv(f'{DIR_INPUT}/filtered_train.csv')  # Set index col to be 0

    # sampled_images = .... (import them)
    # train_df = train_df[sampled_images]

    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(np.float)
    train_df['y'] = train_df['y'].astype(np.float)
    train_df['w'] = train_df['w'].astype(np.float)
    train_df['h'] = train_df['h'].astype(np.float)

    # Define training data to get ffts over training data
    image_ids = train_df['image_id'].unique()
    train_ids = image_ids[:]

    train_df = train_df[train_df['image_id'].isin(train_ids)]

    train_dataset = WheatDataset(train_df, DIR_TRAIN)

    # split the dataset into train and test set
    indices = torch.randperm(len(train_dataset)).tolist()

    print(len(indices))

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    ########################################### generate masks #####################################
    # Each input image is 1024 x 1024, and each bounding box image is also padded
    # with 0s before taking the fft to have the same frequency resolution for the
    # full image and the bounding boxes
    # im_counter = 0
    pad2 = 1024

    win_im = np.hamming(pad2)
    win_im = np.outer(win_im, win_im)
    # win_im = win_im.fill(1.0) # bug
    # win_im = np.ones_like(win_im)

    # plt.figure()
    # plt.imshow(win_im)
    # plt.title('hamming')
    # plt.show()

    wheat_freq = np.zeros((pad2, pad2, 3))
    im_freq = np.zeros((pad2, pad2, 3))

    wheat_freq_cv = np.zeros((pad2, pad2, 3))
    im_freq_cv = np.zeros((pad2, pad2, 3))

    # magnitude_spectrum_cv = np.zeros((pad2, pad2, 3))
    # magnitude_spectrum_wheat_cv = np.zeros((pad2, pad2, 3))

    bg_freq = np.zeros((pad2, pad2, 3))

    for images, targets, image_ids in train_data_loader:
        boxes = targets[0]['boxes']
        image = images[0]

        # get fft of full image in 3 color channels
        for j in range(3):
            # xxx = np.fft.fft2(image[:, :, j] * win_im)
            # print('fft', xxx.shape, xxx)
            img = (image[:, :, j] * win_im)

            im_fft = apply_fft(img, use_fft_np=use_fft_np, use_complex_cv=use_complex_cv)
            im_freq[:, :, j] += im_fft

        # -- Plotting the differences
        # plt.subplot(122), plt.imshow(im_freq_cv, cmap='gray')
        # plt.title('OpenCV'), plt.xticks([]), plt.yticks([])
        #
        # plt.subplot(121), plt.imshow(im_freq, cmap='gray')
        # plt.title('NP package'), plt.xticks([]), plt.yticks([])
        # plt.show()

        # get fft of each bounding box
        bbox_counter = 0
        for i in range(boxes.shape[0]):
            # both boxes and image are np arrays
            coord = (boxes[i, :]).astype(int)

            p = image[coord[1]:coord[3], coord[0]:coord[2], :]

            # Normalize the mean to get rid of DC component
            p = p - np.mean(p)

            # plt.imshow(p)
            # plt.title(str(im_counter+1))
            # plt.pause(0.1)

            p_h = p.shape[0]
            p_w = p.shape[1]

            # define window
            win_wheat = np.outer(np.hamming(p_h), np.hamming(p_w))
            # win_wheat = np.ones_like(win_wheat)
            # pad bounding boxes
            y_pad_b = int(np.ceil((pad2 - p_h) / 2))
            y_pad_a = int(np.floor((pad2 - p_h) / 2))
            x_pad_b = int(np.ceil((pad2 - p_w) / 2))
            x_pad_a = int(np.floor((pad2 - p_w) / 2))
            p_pad = np.pad(p * win_wheat[:, :, None], ((y_pad_b, y_pad_a), (x_pad_b, x_pad_a), (0, 0)))

            # fft over each channel
            if y_pad_b > 0 and x_pad_b > 0:
                for j in range(3):
                    p_fft = apply_fft(p_pad[:, :, j], use_fft_np=use_fft_np, use_complex_cv=use_complex_cv)
                    wheat_freq[:, :, j] += p_fft

                # Plotting the outcome
                # plt.subplot(122), plt.imshow(wheat_freq, cmap='gray')
                # plt.title('wheat NP'), plt.xticks([]), plt.yticks([])
                #
                # plt.subplot(121), plt.imshow(wheat_freq_cv, cmap='gray')
                # plt.title('wheat CV'), plt.xticks([]), plt.yticks([])
                # plt.show()

            bbox_counter += 1

        #         plt.imshow(np.log(np.fft.fftshift(p_fft/np.sum(p_fft))),vmin=-20,vmax=-8)
        #         plt.colorbar()
        #         plt.pause(0.1)

        # im_counter += 1
        # print(im_counter)

        # if im_counter % 10 == 0:
        #     fig, axs = plt.subplots(nrows=2, ncols=2)
        #     axs = axs.ravel()
        #     ax = axs[0]
        #     ax.imshow(wheat_freq[:, :, 0])
        #     ax = axs[1]
        #     ax.imshow(wheat_freq[:, :, 1])
        #     ax = axs[2]
        #     ax.imshow(wheat_freq[:, :, 2])
        #     ax = axs[3]
        #     ax.imshow(wheat_freq)
        #     plt.suptitle(str(im_counter + 1))
        #     plt.show()

        # TODO: STOPPING CRITERION (uncomment the following line to make it faster)
        # if im_counter == 100: break

    # endTime = datetime.datetime.now()
    # duration = endTime - startTime
    # print("This process took: ", duration)

    # if not use_fft_np:
    #     wheat_freq = wheat_freq_cv
    #     im_freq = im_freq_cv

    mask = np.zeros((pad2, pad2, 3))
    for i in range(3):

        plot_wheat = np.log(wheat_freq[:, :, i])
        plot_im = np.log(im_freq[:, :, i])

        print('min(plot_wheat)', np.min(plot_wheat))
        print('min(plot_im)', np.min(plot_im))

        # if printed values all positive, ???
        if np.min(plot_wheat) > 0 and np.min(plot_im) > 0:
            plot_wheat[0, :] = 0
            plot_wheat[:, 0] = 0
            plot_im[0, :] = 0
            plot_im[:, 0] = 0

        plot_wheat = plot_wheat / np.sum(np.abs(plot_wheat))
        plot_im = plot_im / np.sum(np.abs(plot_im))
        print('min(plot_wheat) norm', np.min(plot_wheat))
        print('min(plot_im) norm', np.min(plot_im))

        # # Plotting the outcome
        # plt.subplot(122), plt.imshow(plot_wheat, cmap='gray')
        # plt.title('wheat plot for ' + suffix), plt.xticks([]), plt.yticks([])
        #
        # plt.subplot(121), plt.imshow(plot_im, cmap='gray')
        # plt.title('image plot for ' + suffix), plt.xticks([]), plt.yticks([])
        # plt.show()

        fft_diff = plot_wheat - plot_im
        fft_diff = np.fft.fftshift(fft_diff)

        f_thr = F_THR
        print("THRESHOLD" + str(f_thr))
        mask[:, :, i] = fft_diff > f_thr * 1e-7
        exclude = EXCLUDE
        print("EXCLUDE " + str(exclude))
        mask[:exclude, :, :], mask[-exclude:, :, :] = 0, 0
        mask[:, :exclude, :], mask[:, -exclude:, :] = 0, 0

        # plt.figure()
        # plt.imshow(mask[:, :, i], vmin=-0.2e-7, vmax=2e-7)
        # plt.colorbar()
        # # plt.pause(0.1)
        # plt.title('mask_'+str(i))
        # plt.show()

        mask[:, :, i] = np.fft.ifftshift(mask[:, :, i])
        print('mask', mask.shape)

    mask_file = 'mask_' + suffix + '.mat'
    sio.savemat(mask_file, {'mask': mask})
    ###########################################################################################

    ### load masks
    mask = sio.loadmat(mask_file)['mask']
    mask = np.array(mask).astype(np.float32)
    print('mask', mask.shape)

    # im_counter = 0
    pad2 = 1024

    for idx, (images, target, image_id) in enumerate(train_data_loader):
        image = images[0]
        image_id = image_id[0]
        # get fft of full image in 3 color channels and mask
        im_masked = np.zeros((pad2, pad2, 3))
        for j in range(3):
            if use_fft_np:
                im_masked[:, :, j] = np.real(np.fft.ifft2(np.fft.fft2(image[:, :, j]) * mask[:, :, j]))
            else:
                fft_cv = cv2.dft(np.float32(image[:, :, j]), flags=cv2.DFT_COMPLEX_OUTPUT)
                new_mask = np.zeros((mask.shape[0], mask.shape[1], 2))
                new_mask[:, :, 0] = mask[:, :, j]
                new_mask[:, :, 1] = mask[:, :, j]
                fft_cv = fft_cv * new_mask
                ifft_cv = cv2.idft(fft_cv)
                # ifft_cv = cv2.magnitude(ifft_cv[:, :, 0], ifft_cv[:, :, 1])
                # im_masked[:, :, j] = np.real(ifft_cv)
                im_masked[:, :, j] = ifft_cv[:, :, 0]

        print('np.max(im_masked)', np.max(im_masked), np.min(im_masked))
        # need to normalize to plot
        im_masked = im_masked - np.min(im_masked)
        im_masked = im_masked / np.max(im_masked)

        # print('im_masked', im_masked.shape, im_masked.max(), im_masked.min())

        filename = 'wheat/sampled_fft_images/' + image_id + '.jpg'
        print('filename', idx, filename)
        im_masked *= 255.0
        im_masked = im_masked[:, :, ::-1]
        im_masked = im_masked.astype(np.uint8)
        cv2.imwrite(filename, im_masked)

        # fig, axs = plt.subplots(nrows=1, ncols=2)
        # axs = axs.ravel()
        # ax = axs[0]
        # ax.imshow(image)
        # ax.set_title('im')
        # ax = axs[1]
        # ax.imshow(im_masked)
        # ax.set_title('im_masked')
        # # plt.suptitle(str(im_counter + 1))
        # plt.show()

        # im_counter += 1
        # if im_counter > 100: break

    endTime = datetime.datetime.now()

    print(endTime - startTime)
