import os
import global_constants as gc
import datetime
import re
import numpy as np
import json

files = os.listdir(gc.escaped_dataset)
product_words_lenth = 40
map_cate = json.load(open(gc.map_cate_file, 'r', encoding='utf-8'))
y_lenth = len(map_cate)
key_map_file = gc.vocap_char_file

def make_vocap():
    product_num_map = {}
    model_num_map = {}
    total_num_map = {}
    brands = []
    makers = []
    key_map = {}
    for file in files:
        if file.find('train') == -1:
            continue
        raw = open(os.path.join(gc.escaped_dataset, file), 'r', encoding='utf-8')
        raw_sentences = raw.read().split('\n')
        print(file)
        for idx, raw_sentence in enumerate(raw_sentences):
            if len(raw_sentence) < 1:
                continue
            temp = raw_sentence.split('\t')
            product = temp[0].strip()
            model = temp[1].strip()
            # brand = temp[2].strip()
            # maker = temp[3].strip()
            # label = temp[4].strip()

            words = list(product.replace(' ', ''))
            words.extend(list(model.replace(' ', '')))
            # print(words)
            for word in words:
                if word in key_map.keys():
                    key_map[word] = key_map[word]+1
                else:
                    key_map[word] = 1

    #         num = len(product.split())
    #         if num in product_num_map.keys():
    #             product_num_map[num] = product_num_map[num]+1
    #         else:
    #             product_num_map[num] = 1
    #         num = len(model.split())
    #         if num in model_num_map.keys():
    #             model_num_map[num] = model_num_map[num] + 1
    #         else:
    #             model_num_map[num] = 1
    #         num = len(product.split()) + len(model.split())
    #         if num in total_num_map.keys():
    #             total_num_map[num] = total_num_map[num]+1
    #         else:
    #             total_num_map[num] = 1
    # for key in product_num_map.keys():
    #     print(key, product_num_map[key])
    # for key in model_num_map.keys():
    #     print(key, model_num_map[key])
    # for key in total_num_map.keys():
    #     print(key, total_num_map[key])
    # words = list(set(words))

    with open(key_map_file, 'w', encoding='utf-8') as output_file :
        output_file.write('UNK' + '\n')
        output_file.writelines("\n".join(list(key_map.keys())))
        for key in key_map.keys():
            if key_map[key] != 1:
                output_file.write(key + '\n')


def make_train_dataset():
    key_map = open(key_map_file, 'r', encoding='utf-8').read().split('\n')
    change_map = {}
    for key_idx, key in enumerate(key_map):
        if len(key) == 0:
            continue
        change_map[key] = key_idx
    os.makedirs(gc.classification_training_dataset_dir, exist_ok=True)

    total_idx = 0
    for idx_file, file in enumerate(files):
        if file.find('train') == -1:
            continue
        print(file)
        items = open(os.path.join(gc.escaped_dataset, file), 'r', encoding='utf-8').read().split('\n')
        items.pop()
        print(len(items))
        x = np.ndarray([len(items), product_words_lenth], dtype=np.uint16)
        # 57, 552, 3190, 404 4215
        y = np.ndarray([len(items), 5], dtype=np.uint16)
        for idx, item in enumerate(items):
            if len(item) < 1:
                continue
            # real_idx = idx_file*1000000 + idx
            if total_idx % 10000 == 0:
                print(total_idx)

            x_total = np.zeros(product_words_lenth, dtype=np.uint32)
            y_total = np.zeros(5, dtype=np.uint32)

            components = item.split('\t')
            product = components[0].strip()
            model = components[1].strip()
            label = components[5].strip()

            temp_product = list(product.replace(' ', ''))
            temp_product.extend(list(model.replace(' ', '')))

            # temp_model = (model.split())

            for word_idx, word in enumerate(temp_product):
                if word_idx == product_words_lenth:
                    break
                x_total[word_idx] = change_map[word]

            labels = label.split('>')
            for label_idx, one_label in enumerate(labels):
                if one_label != '-1':
                    y_total[label_idx] = int(one_label) - 1
                else:
                    y_total[label_idx] = 0
            y_total[4] =  map_cate[label]
            x[idx] = x_total
            y[idx] = y_total
            total_idx +=1

        out_file_name = file.replace('.txt', '')
        np.save(os.path.join(gc.classification_training_dataset_dir, 'embedding_x_' + out_file_name), x)
        np.save(os.path.join(gc.classification_training_dataset_dir, 'embedding_y_' + out_file_name), y)

def make_val_dataset():
    key_map = open(key_map_file, 'r', encoding='utf-8').read().split('\n')
    change_map = {}
    for key_idx, key in enumerate(key_map):
        if len(key) == 0:
            continue
        change_map[key] = key_idx
    os.makedirs(gc.classification_training_dataset_dir, exist_ok=True)

    total_idx = 0
    for idx_file, file in enumerate(files):
        if file.find('train') != -1:
            continue
        print(file)
        items = open(os.path.join(gc.escaped_dataset, file), 'r', encoding='utf-8').read().split('\n')
        items.pop()
        print(len(items))
        x = np.ndarray([len(items), product_words_lenth], dtype=np.uint16)
        pids = []

        for idx, item in enumerate(items):
            if len(item) < 1:
                continue
            # real_idx = idx_file*1000000 + idx
            if total_idx % 100000 == 0:
                print(total_idx)

            x_total = np.zeros(product_words_lenth, dtype=np.uint32)

            components = item.split('\t')
            product = components[0].strip()
            model = components[1].strip()
            pid = components[4].strip()

            temp_product = list(product.replace(' ', ''))
            temp_product.extend(list(model.replace(' ', '')))

            # temp_model = (model.split())

            for word_idx, word in enumerate(temp_product):
                if word_idx == product_words_lenth:
                    break
                if word in change_map.keys():
                    x_total[word_idx] = change_map[word]
                else:
                    x_total[word_idx] = change_map['UNK']

            x[idx] = x_total
            total_idx +=1
            pids.append(pid)

        out_file_name = file.replace('.txt', '')
        np.save(os.path.join(gc.classification_dev_dataset_dir, 'embedding_x_' + out_file_name), x)
        open(os.path.join(gc.classification_dev_dataset_dir, (out_file_name + '_pid.txt')), 'w', encoding='utf-8').writelines("\n".join(pids))

if __name__ == '__main__':

    ###########################
    # make_vocap()
    ###########################
    make_val_dataset()
    ###########################

