import pandas as pd
import numpy as np
import re
from shutil import copyfile
import os
import shutil
import subprocess
from argparse import ArgumentParser
import torch

torch.cuda.empty_cache()

parser = ArgumentParser()
parser.add_argument("--n_images", default=50, help="number of images to sample")
parser.add_argument("--batch", default=1, help="Batch size for yolo")
parser.add_argument("--epochs", default=15, help="Number of epochs to perform")
parser.add_argument("--threshold", default=0.2, help="FFT threshold")
parser.add_argument("--exclude", default=120, help="FFT high-pass threshold")

args = parser.parse_args()
args = vars(args)

IMAGE_SIZE = int(args['n_images'])

BATCH_SIZE = args['batch']

EPOCHS = args['epochs']

F_THR = args['threshold']

EXCLUDE = args['exclude']

cmd = 'python yolov5/train.py --img 1024 --batch ' + str(BATCH_SIZE) + ' --epochs ' + str(
    EPOCHS) + ' --data wheat_head_data' \
              '.yaml --weights yolov5s.pt' \
              ' --nosave --cache'

df = pd.read_csv('wheat/train.csv')
df['x'] = -1
df['y'] = -1
df['w'] = -1
df['h'] = -1

img_width = 1024
img_height = 1024


def x_center(df):
    return df.xmin + (df.width / 2)


def y_center(df):
    return df.ymin + (df.height / 2)


def w_norm(df):
    return df / img_width


def h_norm(df):
    return df / img_height


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


def empty_files(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def prepare_data(sampled_images, images_folder, train_size, validation_size, test_size):
    """

    :param sampled_images: image ids
    :return: creates folders with preprocessed data ready to be analysed from yolov
    """
    N = len(sampled_images)
    train_splt_idx = int(train_size * N)
    validation_split_idx = train_splt_idx + int(validation_size * N)
    for j, image in enumerate(sampled_images):
        image_data = df[df['image_id'] == image]
        filtered_image_data = image_data[['class', 'x_center_norm', 'y_center_norm', 'width_norm', 'height_norm']]
        title = image + '.txt'
        if j < train_splt_idx:  # Then this is a training image
            filtered_image_data.to_csv('train_data/labels/train/' + title, header=None, index=None, sep=' ', mode='a')
            copyfile(images_folder + image + '.jpg', 'train_data/images/train/' + image + '.jpg')
            # Also add the respective picture
        elif train_splt_idx <= j < validation_split_idx:
            filtered_image_data.to_csv('train_data/labels/val/' + title, header=None, index=None, sep=' ', mode='a')
            copyfile(images_folder + image + '.jpg', 'train_data/images/val/' + image + '.jpg')
        elif j >= validation_split_idx:
            filtered_image_data.to_csv('train_data/labels/test/' + title, header=None, index=None, sep=' ', mode='a')
            copyfile(images_folder + image + '.jpg', 'train_data/images/test/' + image + '.jpg')


def last_pt(path):
    m = -1
    dirs = next(os.walk(path))
    dirs = dirs[1]
    for dir in dirs:
        if isinstance(dir, str):
            splt = dir.split('exp')
            if splt[1]:
                run = int(splt[1])
                m = max(run, m)
    return path + '/exp' + str(m) + '/weights/last.pt'


df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
# df.drop(columns=['bbox'], inplace=True)
df['xmin'] = df['x'].astype(float)
df['ymin'] = df['y'].astype(float)
df['width'] = df['w'].astype(float)
df['height'] = df['h'].astype(float)

df['x_center'] = df.apply(x_center, axis=1)
df['y_center'] = df.apply(y_center, axis=1)

df['x_center_norm'] = df['x_center'].apply(w_norm)
df['y_center_norm'] = df['y_center'].apply(h_norm)
df['width_norm'] = df['width'].apply(w_norm)
df['height_norm'] = df['height'].apply(h_norm)
df['class'] = 0

images = df['image_id'].unique()

# Folders to empty
folders = ['train_data/images/train', 'train_data/images/val', 'train_data/labels/train', 'train_data/labels/val',
           'train_data/labels/test/', 'train_data/images/test',
           'wheat/sampled_images', 'wheat/sampled_fft_images']

# Delete all contents of folder
empty_files(folders)

sampled_images = np.random.choice(images, IMAGE_SIZE)
for image_id in sampled_images:
    copyfile('wheat/train/' + image_id + '.jpg', 'wheat/sampled_images/' + image_id + '.jpg')

# Prepare data for the yolo
prepare_data(sampled_images, images_folder='wheat/sampled_images/', train_size=0.7, validation_size=0.2, test_size=0.1)

process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Now test
weights = last_pt('runs/train')
cmd = 'python yolov5/test.py --weights ' + weights + ' --data wheat_head_data.yaml --img 1024 --iou 0.65 --batch 2'
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

torch.cuda.empty_cache()

filtered_image_df = df[df['image_id'].isin(sampled_images)]  # Wheat preprocessing needs it
# print('Unique images: ', len(filtered_image_df['image_id'].unique())) Uncomment to verify length (should be the number of random samples taken)
filtered_image_df.to_csv('wheat/filtered_train.csv')

cmd = 'python wheat_preprocessing.py --threshold ' + str(F_THR) + ' --exclude ' + str(EXCLUDE)
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Empty folders
folders = ['train_data/images/train', 'train_data/images/val', 'train_data/labels/train', 'train_data/labels/val',
           'train_data/labels/test', 'train_data/images/test']
# Delete all contents of these folders
empty_files(folders)

# Prepare data for the yolo
prepare_data(sampled_images, images_folder='wheat/sampled_fft_images/', train_size=0.7, validation_size=0.2,
             test_size=0.1)

# Now run yolov5 on the fft images
cmd = 'python yolov5/train.py --img 1024 --batch ' + str(BATCH_SIZE) + ' --epochs ' + str(
    EPOCHS) + ' --data wheat_head_data' \
              '.yaml --weights yolov5s.pt' \
              ' --nosave --cache'
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Now run the testing
weights = last_pt('runs/train')
cmd = 'python yolov5/test.py --weights ' + weights + ' --data wheat_head_data.yaml --img 1024 --iou 0.65 --batch 2'
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Empty folders
empty_files(folders)

# #!TODO run yolo on it and compare (somehow) the results