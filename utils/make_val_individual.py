import os
import global_constants as gcon
import datetime
import re
import numpy as np
import json


dd = '요금제|개월|개|부|종|박스|상자|gb|box|km|oz|kg|mg|ml|ea|㎡|cm|mm|l|m|g|인치|inch|%'
files = os.listdir(os.path.join(gcon.base_data_dir, 'dev'))
words_lenth = 40

map_cate = json.load(open('../data/label.json', 'r', encoding='utf-8'))
y_lenth = len(map_cate)
key_map_file = os.path.join(gcon.embedding_model_file_path, 'vocab_char.txt')
key_map = open(key_map_file, 'r', encoding='utf-8').read().split('\n')

# 그냥 list index로 하면 엄청 느링
change_map = {}
for key_idx, key in enumerate(key_map):
    change_map[key] = key_idx

print(key_map[:3])
os.makedirs(gcon.tr_fl_classification_dev_dataset_dir, exist_ok=True)

def main_x_y():
    total_idx = 0
    # x = np.ndarray([len_itmes, words_lenth], dtype=np.uint16)
    # y = np.ndarray([len_itmes], dtype=np.uint16)
    for idx_file, file in enumerate(files):
        if file.find('dev01.txt') == -1:
            continue
        print(file)
        items = open(os.path.join(gcon.base_data_dir, 'dev', file), 'r', encoding='utf-8').read().split('\n')
        items.pop()
        x = np.ndarray([len(items), words_lenth], dtype=np.uint16)
        pids = []
        for idx, item in enumerate(items):
            if len(item) < 1:
                continue
            # real_idx = idx_file*1000000 + idx
            if total_idx % 10000 == 0:
                print(total_idx)

            x_product = np.zeros(words_lenth, dtype=np.uint32)

            components = item.split('\t')
            product = components[0].strip()
            model = components[1].strip()
            re_product = escape_units(product)
            re_model = escape_units(model)

            pid = components[4].strip()

            # temp_x = re_product.split()
            # temp_x.extend(re_model.split())
            temp_x = list(re_product.replace(' ', ''))
            temp_x.extend(list(re_model.replace(' ', '')))

            for word_idx, word in enumerate(temp_x):
                if word_idx == words_lenth:
                    break
                if word in change_map.keys():
                    x_product[word_idx] = change_map[word]
                else:
                    x_product[word_idx] = change_map['UNK']

            # print(product)
            # print(x_product)

            x[idx] = x_product
            pids.append(pid)
            total_idx +=1


        out_file_name = file.replace('.txt', '')
        np.save(os.path.join(gcon.tr_fl_classification_dev_dataset_dir, 'embedding_x_' + out_file_name), x)
        open(os.path.join(gcon.tr_fl_classification_dev_dataset_dir, 'dev01_pid.txt'), 'w', encoding='utf-8').writelines("\n".join(pids))

def escape_units(raw_sentence):

    if len(raw_sentence) < 3:
        return raw_sentence
    # 개, 원, 팩, g, G, ml, kg, 장 , 병 ,지, 봉, 캡슐, x, cm, l (리터)
    escaped_sentence = raw_sentence.lower()
    escaped_sentence = re.sub(
        '[a-z|0-9]+[_|-]+[_a-z0-9|-]{1,}',
        " ", escaped_sentence)

    escaped_sentence = re.sub(
        # "[0-9.,]+\s?(박스|상자|box|km|l|ml|oz|g|kg|mg|ea|㎡|cm|mm|m||인치|inch)+(\s?|$)",
        # "[0-9.,]+(" + dd + "){1}(\s|$)",
        "[0-9.,]+(" + dd + "){1}",
        " ", escaped_sentence)

    escaped_sentence = re.sub("[^0-9가-힣a-z]", " ", escaped_sentence)

    # 숫자로만 제거...
    escaped_sentence = re.sub(
        '\s[0-9]{3,}\s?',
        " ", escaped_sentence)

    escaped_sentence = escaped_sentence.split()

    result = []
    for word in escaped_sentence:
        word = word.strip()
        if len(word) < 2:
            continue

            # temp = re.sub('[가-힣]', '', word)
        if len(re.sub('[가-힣]', '', word)) == 0:
            result.append(word)
        elif len(re.sub('[a-z]', '', word)) == 0:
            result.append(word)

        elif len(re.sub('[a-z0-9]', '', word)) == 0:
            continue

        else:
            temps = re.sub('[^가-힣]', ' ', word)
            for temp in temps.split():
                if len(temp) > 1:
                    result.append(temp)

    return " ".join(result)

if __name__ == '__main__':
    main_x_y()

