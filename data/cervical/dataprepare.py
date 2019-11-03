# -*- coding: utf-8 -*-

import os
import random
import shutil

SOURCE_PATH = r'D:\Project\Biopsy'
TARGET_ROOT_PATH = r'C:\Users\FlyingMoon\PycharmProjects\FNet\data\cervical'
TARGET_ANNO_DIR = r'anno'
TARGET_IMAGE_DIR = r'image'
CATEGORY_NAMES = ['Cancer', 'HSIL', 'LSIL', 'No_Biopsy']
REAGENT_NAMES = ['VIA3', 'VILI']
ANNO_FILE_NAMES = ['train.txt', 'val.txt', 'test.txt']
PROPOTIONS = [0.6, 0.1, 0.3]

if __name__ == '__main__':

    if not os.path.exists(os.path.join(TARGET_ROOT_PATH, TARGET_IMAGE_DIR)):
        os.mkdir(os.path.join(TARGET_ROOT_PATH, TARGET_IMAGE_DIR))

    categories = []
    samples_list = []
    category_nums = 0
    sample_nums = 0

    # for category in os.listdir(SOURCE_PATH):
    for category in CATEGORY_NAMES:
        print(category)
        categories.append(category)
        for patient_code in os.listdir(os.path.join(SOURCE_PATH, category)):
            print(patient_code)
            samples_list.append((patient_code, category_nums))
            if not os.path.exists(os.path.join(TARGET_ROOT_PATH, TARGET_IMAGE_DIR, patient_code)):
                os.mkdir(os.path.join(TARGET_ROOT_PATH, TARGET_IMAGE_DIR, patient_code))
            for reagent_name in REAGENT_NAMES:
                print(reagent_name)
                if not os.path.exists(
                        os.path.join(TARGET_ROOT_PATH, TARGET_IMAGE_DIR, patient_code, reagent_name + '.jpg')):
                    shutil.copy(os.path.join(SOURCE_PATH, category, patient_code, reagent_name + '.jpg'),
                                os.path.join(TARGET_ROOT_PATH, TARGET_IMAGE_DIR, patient_code, reagent_name + '.jpg'))
            sample_nums += 1
        category_nums += 1

    with open(os.path.join(TARGET_ROOT_PATH, TARGET_ANNO_DIR, 'category_dict.txt'), 'w') as f:
        for i, category in enumerate(categories):
            f.write(str(i) + ' ' + category + '\n')

    print('Copy Completed')

    assert sum(PROPOTIONS) <= 1.0

    random.shuffle(samples_list)

    print('Shuffle Completed')

    cur_propotion = 0
    sample_range = [0, 0]

    for i, anno_file_name in enumerate(ANNO_FILE_NAMES):
        cur_propotion += PROPOTIONS[i]
        sample_range[0] = sample_range[1]
        sample_range[1] = int(sample_nums * cur_propotion) - 1
        print(anno_file_name)
        with open(os.path.join(TARGET_ROOT_PATH, TARGET_ANNO_DIR, anno_file_name), 'w') as f:
            for sample_no in range(sample_range[0], sample_range[1]):
                print(samples_list[sample_no][0], samples_list[sample_no][1])
                f.write(samples_list[sample_no][0] + ' ' + str(samples_list[sample_no][1]) + '\n')

    print('Annotate Completed')

    print('All Completed')
