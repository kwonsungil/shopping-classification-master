import h5py
import json
import numpy as np
import os
import global_constants as gc

def split():
    data_dir = gc.raw_dataset
    output_dir = gc.prerocess_dataset
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gc.image_dataset, exist_ok=True)
    # chunk_files = ['train.chunk.09']
    chunk_files = os.listdir(gc.raw_dataset)

    for chunk_file in chunk_files:
        if chunk_file.find('chunk') == -1:
            continue
        print(chunk_file)
        datas = h5py.File(os.path.join(data_dir, chunk_file), 'r')
        output_file = os.path.join(output_dir, chunk_file.replace('chunk.', '') + '.txt')
        output_np_file = os.path.join(gc.image_dataset, chunk_file.replace('chunk.', ''))

        mode = chunk_file.split('.')[0]
        datas = datas[mode]
        size = len(datas['product'])
        result_feat = np.zeros([size, 2048], dtype=np.float32)

        with open(output_file, 'w', encoding='utf-8') as output:
            for idx in range(size):
                if idx % 1000 == 0 and idx != 0:
                    print(idx)
                    break
                temp = []
                temp.append(datas['product'][idx].decode('utf-8').replace('\t', '').strip())
                temp.append(datas['model'][idx].decode('utf-8').replace('\t', '').strip())
                temp.append(datas['brand'][idx].decode('utf-8').replace('\t', '').strip())
                temp.append(datas['maker'][idx].decode('utf-8').replace('\t', '').strip())
                temp.append(datas['pid'][idx].decode('utf-8').replace('\t', '').strip())
                temp.append(str(datas['price'][idx]).replace('\t', '').strip())
                label = str(datas['bcateid'][idx]) + '>' + str(datas['mcateid'][idx]) + '>' + str(datas['scateid'][idx]) + '>' + str(datas['dcateid'][idx])
                temp.append(label.strip())
                # print(datas['img_feat'][idx])
                result_feat[idx] = datas['img_feat'][idx]
                # print("\t".join(temp))
                output.write("\t".join(temp) + '\n')
            np.save(output_np_file, result_feat)

if __name__ == '__main__':
    split()
