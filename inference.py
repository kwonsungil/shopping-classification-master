from model.model_v9_1 import kakao
import numpy as np
import os, json
import global_constants as gc
import math


def last_layer():
    results_cate = []
    for batch_num in range(num_batches_per_epoch):
        print(batch_num, num_batches_per_epoch)
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, total_lenth)
        logits5, logits6 = kakao.predict_last_layer(x[start_index:end_index], images[start_index:end_index])
        # results_cate.extend(kakao.predict_last_layer(x[start_index:end_index], images[start_index:end_index]))
        logit = np.copy(logits6)
        logit[logits5[:,1] > logits6[:,1]] = logits5[logits5[:,1] > logits6[:,1]]
        results_cate.extend(logit[:,0])

    print(len(results_cate))
    for result_idx, pid in enumerate(pids):
        a = results_cate[result_idx]
        cate = reverse_cate[a]
        # cate = reverse_cate[results_cate[result_idx]]
        result_file.write(pid + '\t' + '\t'.join(cate.split('>')) + '\n')
    result_file.close()

if __name__ == '__main__':
    batch_size = 5000
    key_map = open(gc.vocap_char_file, 'r', encoding='utf-8').read().split('\n')
    change_map = {}
    for key_idx, key in enumerate(key_map):
        change_map[key] = key_idx

    map_cate = json.load(open(gc.map_cate_file, 'r', encoding='utf-8'))
    reverse_cate = {}
    for key in map_cate.keys():
        reverse_cate[map_cate[key]] = key

    total_world = len(key_map)
    print(total_world)
    kakao = kakao(total_world, False)

    files = ['test01', 'test02']
    # files = ['dev01']
    for file in files:
        print(file)
        output_file = file + '_result.txt'
        result_file = open(output_file, 'w')
        pids = open(os.path.join(gc.classification_dev_dataset_dir, file + '_pid.txt'), 'r', encoding='utf-8').read().split('\n')
        total_lenth = len(pids)

        x = np.load(os.path.join(gc.classification_dev_dataset_dir, 'embedding_x_' + file + '.npy'))
        images = np.load(os.path.join(gc.image_dataset, file + '.npy'))

        num_batches_per_epoch = math.ceil(total_lenth / batch_size)
        last_layer()

