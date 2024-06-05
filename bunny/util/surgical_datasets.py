
import os
import json
import pandas as pd
from tqdm import tqdm

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, load_image_from_path, \
    get_model_name_from_path

from torch.utils.data import Dataset, DataLoader

class MedVQAClassification(Dataset):
    '''
        VQA-MED 19 dataloader
        datafloder = data floder location
        img_floder = image floder location
        cat        = category (1/2/3)
        patch_size = patch size of image, which also determines token size (1/2/3/4/5)
        validation = if validation, load val.txt, else load train.txt (True / False)
    '''

    def getOptions(self):
        return self.labels


    def __init__(self, datafloder, imgfloder, image_processor, model, cat, patch_size=4, dataType='train'):

        self.data_floder_loc = datafloder
        self.img_floder_loc = imgfloder

        self.image_processor = image_processor
        self.model = model
        if cat == 'cat1':
            if dataType == 'val':
                self.file_name = 'QAPairsByCategory/C1_Modality_val.txt'
            elif dataType == 'train':
                self.file_name = 'QAPairsByCategory/C1_Modality_train.txt'
            elif dataType == 'test':
                self.file_name = 'QAPairsByCategory/C1_Modality_test.txt'
            else:
                raise ValueError('Invalid dataType')  # 抛出错误
            self.labels = ['cta - ct angiography', 'no', 'us - ultrasound', 'xr - plain film', 'noncontrast', 'yes',
                           't2', 'ct w/contrast (iv)', 'mr - flair', 'mammograph', 'ct with iv contrast',
                           'gi and iv', 't1', 'mr - t2 weighted', 'mr - t1w w/gadolinium', 'contrast', 'iv',
                           'an - angiogram', 'mra - mr angiography/venography', 'nm - nuclear medicine',
                           'mr - dwi diffusion weighted',
                           'ct - gi & iv contrast', 'ct noncontrast', 'mr - other pulse seq.',
                           'ct with gi and iv contrast', 'flair', 'mr - t1w w/gd (fat suppressed)', 'ugi - upper gi',
                           'mr - adc map (app diff coeff)',
                           'bas - barium swallow', 'pet - positron emission', 'mr - pdw proton density',
                           'mr - t1w - noncontrast', 'be - barium enema', 'us-d - doppler ultrasound', 'mr - stir',
                           'mr - flair w/gd',
                           'ct with gi contrast', 'venogram', 'mr t2* gradient,gre,mpgr,swan,swi', 'mr - fiesta',
                           'ct - myelogram', 'gi', 'sbft - small bowel', 'pet-ct fusion']
        elif cat == 'cat2':
            if dataType == 'val':
                self.file_name = 'QAPairsByCategory/C2_Plane_val.txt'
            elif dataType == 'train':
                self.file_name = 'QAPairsByCategory/C2_Plane_train.txt'
            elif dataType == 'test':
                self.file_name = 'QAPairsByCategory/C2_Plane_test.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = ['axial', 'longitudinal', 'coronal', 'lateral', 'ap', 'sagittal', 'mammo - mlo', 'pa',
                           'mammo - cc', 'transverse', 'mammo - mag cc', 'frontal', 'oblique', '3d reconstruction',
                           'decubitus', 'mammo - xcc']
        elif cat == 'cat3':
            if dataType == 'val':
                self.file_name = 'QAPairsByCategory/C3_Organ_val.txt'
            elif dataType == 'train':
                self.file_name = 'QAPairsByCategory/C3_Organ_train.txt'
            elif dataType == 'test':
                self.file_name = 'QAPairsByCategory/C3_Organ_test.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = ['lung, mediastinum, pleura', 'skull and contents', 'genitourinary', 'spine and contents',
                           'musculoskeletal', 'heart and great vessels', 'vascular and lymphatic', 'gastrointestinal',
                           'face, sinuses, and neck', 'breast']
        elif cat == 'all':
            if dataType == 'test':
                self.file_name = 'answers.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = ['cta - ct angiography', 'no', 'us - ultrasound', 'xr - plain film', 'noncontrast', 'yes',
                           't2', 'ct w/contrast (iv)', 'mr - flair', 'mammograph', 'ct with iv contrast',
                           'gi and iv', 't1', 'mr - t2 weighted', 'mr - t1w w/gadolinium', 'contrast', 'iv',
                           'an - angiogram', 'mra - mr angiography/venography', 'nm - nuclear medicine',
                           'mr - dwi diffusion weighted',
                           'ct - gi & iv contrast', 'ct noncontrast', 'mr - other pulse seq.',
                           'ct with gi and iv contrast', 'flair', 'mr - t1w w/gd (fat suppressed)', 'ugi - upper gi',
                           'mr - adc map (app diff coeff)',
                           'bas - barium swallow', 'pet - positron emission', 'mr - pdw proton density',
                           'mr - t1w - noncontrast', 'be - barium enema', 'us-d - doppler ultrasound', 'mr - stir',
                           'mr - flair w/gd',
                           'ct with gi contrast', 'venogram', 'mr t2* gradient,gre,mpgr,swan,swi', 'mr - fiesta',
                           'ct - myelogram', 'gi', 'sbft - small bowel', 'pet-ct fusion',

                           'axial', 'longitudinal', 'coronal', 'lateral', 'ap', 'sagittal', 'mammo - mlo', 'pa',
                           'mammo - cc', 'transverse', 'mammo - mag cc', 'frontal', 'oblique', '3d reconstruction',
                           'decubitus', 'mammo - xcc',

                           'lung, mediastinum, pleura', 'skull and contents', 'genitourinary', 'spine and contents',
                           'musculoskeletal', 'heart and great vessels', 'vascular and lymphatic', 'gastrointestinal',
                           'face, sinuses, and neck', 'breast'
                           ]
        self.patch_size = patch_size

        self.vqas = []
        print(self.data_floder_loc, self.file_name)
        file_data = open((os.path.join(self.data_floder_loc, self.file_name)), "r")
        lines = [line.strip("\n") for line in file_data if line != "\n"]
        file_data.close()
        # print("lines:",lines)
        for line in lines: self.vqas.append([line])

        print('Total question: %.d' % len(lines))

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):

        # img
        self.image_path = os.path.join(self.img_floder_loc, self.vqas[idx][0].split('|')[0]) + '.jpg'
        image = load_image_from_path(self.image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        # question and answer
        # print("self.vqas[idx][0].split('|')",self.vqas[idx][0].split('|'))
        question = self.vqas[idx][0].split('|')[2]
        answer = self.vqas[idx][0].split('|')[3]

        # print("1:/n", self.vqas[idx][0].split('|')[0], "2:/n", image_tensor, "3:/n", question, "4:/n", answer)

        return self.vqas[idx][0].split('|')[0], image_tensor, question, answer
