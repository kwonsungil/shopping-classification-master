import os
import datetime
import global_constants as gc
import re

dd = '요금제|개월|개|부|종|박스|상자|gb|box|km|oz|kg|mg|ml|ea|㎡|cm|mm|l|m|g|인치|inch|%'


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


def process_dataset():
    input_dir = gc.prerocess_dataset
    output_dir = gc.escaped_dataset
    os.makedirs(output_dir, exist_ok=True)

    total_count = 0
    files = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file in files:
        print(file)
        items = open(os.path.join(input_dir, file), 'r', encoding="utf-8").read().split('\n')
        output = open(os.path.join(output_dir, file), 'w', encoding='utf-8')

        for item in items:
            if len(item) == 0:
                continue
            label = item.split('\t')[6]
            escaped_sentence = item.split('\t')[0].strip()
            model = item.split('\t')[1].strip()
            brand = item.split('\t')[2].strip()
            maker = item.split('\t')[3].strip()
            pid = item.split('\t')[4].strip()

            refined_escaped_sentence = escape_units(escaped_sentence)
            # print(escaped_sentence)
            # print(refined_escaped_sentence)
            # print('--------------------------')
            refined_model = escape_units(model)
            output.write(
                refined_escaped_sentence + '\t' + refined_model + '\t' + brand + '\t' + maker + '\t' + pid + '\t' + label + '\n')
            total_count += 1

        print(file + " = %d" % len(items))
        output.close()
    print("\nTotal Count = %d\n" % total_count)



if __name__ == '__main__':
    process_dataset()
