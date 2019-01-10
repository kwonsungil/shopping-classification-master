from utils.data_split import split
from utils.make_train_dataset import make_vocap, make_train_dataset, make_val_dataset
from utils.refine_and_tag_for_char import process_dataset
import os
import global_constants as gc
import numpy as np
import math, random
from model.model_v9_1 import kakao

if __name__ == '__main__':
    if not os.path.exists(gc.classification_training_dataset_dir) or len(os.listdir(gc.classification_training_dataset_dir)) == 0:
        print('start split!')
        split()
        print('start process_dataset!')
        process_dataset()
        #print('start make_vocap!')
        #make_vocap()
        print('start make_train_dataset!')
        make_train_dataset()
        print('start make_val_dataset!')
        make_val_dataset()
    batch_size = 4000
    files = ['train.01', 'train.02', 'train.03', 'train.04', 'train.05', 'train.06', 'train.07', 'train.08', 'train.09']
    x_y_dir = gc.classification_training_dataset_dir
    key_map = open(gc.vocap_char_file, 'r', encoding='utf-8').read().split('\n')
    total_world = len(key_map)
    print(total_world)

    total_x = []
    total_y = []
    for file in files:
        x_filename = 'embedding_x_' + file + '.npy'
        y_filename = 'embedding_y_' + file + '.npy'
        # x = np.load(os.path.join('C:', 'data', x_filename))
        # y = np.load(os.path.join('C:', 'data', y_filename))
        x = np.load(os.path.join(gc.classification_training_dataset_dir, x_filename))
        y = np.load(os.path.join(gc.classification_training_dataset_dir, y_filename))
        total_x.append(x)
        total_y.append(y)

    del x
    del y

    kakao = kakao(total_world, True)
    # files.reverse()
    for epoch in range(80):
        for file_idx, file in enumerate(files):
            image_filename = file + '.npy'
            # images = np.load(os.path.join('C:', 'data', image_filename))
            images = np.load(os.path.join(gc.image_dataset, image_filename))

            # print(x.shape)
            # print(y.shape)
            print(images.shape)

            total_lenth = len(total_x[file_idx])
            random_idx = list(range(total_lenth))
            random.shuffle(random_idx)
            num_batches_per_epoch = math.ceil(total_lenth / batch_size)

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, total_lenth)
                kakao.train(epoch, batch_num, total_x[file_idx][random_idx[start_index:end_index]],
                            images[random_idx[start_index:end_index]],
                            total_y[file_idx][random_idx[start_index:end_index]])
