import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, process_images,load_image_from_path, \
    get_model_name_from_path

import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from bunny.util.data_utils import EndoVis18VQAClassification




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
        self.cat_name_map={'cat1':'modality','cat2':'plane','cat3':'organ','cat4':'abnormality'}

        self.image_processor = image_processor
        self.model = model
        self.cat = cat




        self.labels1=['venogram', 'pet - positron emission', 'be - barium enema', 'no', 'an - angiogram',
         'mr - other pulse seq.', 'mr - dwi diffusion weighted', 'sbft - small bowel', 
         'mr - adc map (app diff coeff)', 't1', 'bas - barium swallow', 'xr - plain film',
        'yes', 'mr - fiesta', 'ct noncontrast', 'mr - flair w/gd', 'mra - mr angiography/venography',
        'contrast', 'flair', 'mr - t1w - noncontrast', 'nm - nuclear medicine', 'us-d - doppler ultrasound',
        'noncontrast', 'ct - myelogram', 'ct with iv contrast', 'us - ultrasound', 'mammograph', 
        'ct with gi and iv contrast', 'gi', 'mr - t1w w/gadolinium', 'mr - t2 weighted', 
        'ct with gi contrast', 'mr - flair', 'ct w/contrast (iv)', 'mr t2* gradient,gre,mpgr,swan,swi',
        'iv', 'mr - stir', 'mr - pdw proton density', 'ct - gi & iv contrast', 'ugi - upper gi', 
        'mr - t1w w/gd (fat suppressed)', 'cta - ct angiography', 'gi and iv', 't2']
        
        self.labels2=['oblique', 'coronal', 'axial', 'frontal', 'pa', 'lateral', 'sagittal', 'mammo - cc',
                       '3d reconstruction', 'decubitus', 'mammo - mag cc', 'mammo - mlo', 'ap',
                         'transverse', 'longitudinal']

        self.labels3=['lung, mediastinum, pleura', 'spine and contents', 'genitourinary', 'breast',
                       'skull and contents', 'face, sinuses, and neck', 'gastrointestinal',
                         'heart and great vessels', 'musculoskeletal', 'vascular and lymphatic']

        lables_test=['no', 'oblique', 'face, sinuses, and neck#skull and contents', 't2', 'coronal',
                      'transverse', 'breast', 'gastrointestinal', 'us-d - doppler ultrasound', 'yes',
                        'ct w/contrast (iv)', 'mr - adc map (app diff coeff)', 'mr - flair', 'ct noncontrast',
                        'contrast', 'sbft - small bowel', 't1', 'axial', 'sagittal', 'face, sinuses, and neck',
                        'heart and great vessels', 'ugi - upper gi', 'mr - t2 weighted', 'us - ultrasound',
                        'mr - t1w w/gadolinium', 'skull and contents', 'mammo - mlo', 'ap', 'mammograph',
                        'gastrointestinal#genitourinary#spine and contents#musculoskeletal',
                        'gi and iv', 'ct with iv contrast', 'iv', 'lung, mediastinum, pleura',
                        'gastrointestinal#lung, mediastinum, pleura', 'spine and contents',
                        'lateral', 'vascular and lymphatic', 'musculoskeletal', 'pa', 'mammo - cc',
                        'heart and great vessels#lung, mediastinum, pleura', 'lung, mediastinum, pleura#spine and contents', 
                        'heart and great vessels#lung, mediastinum, pleura#spine and contents', 'ct with gi and iv contrast',
                        'longitudinal', 'skull and contents#spine and contents', '3d reconstruction', 'frontal',
                        'noncontrast', 'genitourinary', 'xr - plain film', 'flair', 'cta - ct angiography',
                            'ct - gi & iv contrast', 'an - angiogram']

        if cat == 'cat1':
            if dataType == 'val':
                self.file_name = 'QAPairsByCategory/C1_Modality_val.txt'
            elif dataType == 'train':
                self.file_name = 'QAPairsByCategory/C1_Modality_train.txt'
            elif dataType == 'test':
                self.file_name = 'answers.txt'
            else:
                raise ValueError('Invalid dataType')  # 抛出错误
            self.labels = self.labels1
        elif cat == 'cat2':
            if dataType == 'val':
                self.file_name = 'QAPairsByCategory/C2_Plane_val.txt'
            elif dataType == 'train':
                self.file_name = 'QAPairsByCategory/C2_Plane_train.txt'
            elif dataType == 'test':
                self.file_name = 'answers.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = self.labels2
        elif cat == 'cat3':
            if dataType == 'val':
                self.file_name = 'QAPairsByCategory/C3_Organ_val.txt'
            elif dataType == 'train':
                self.file_name = 'QAPairsByCategory/C3_Organ_train.txt'
            elif dataType == 'test':
                self.file_name = 'answers.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = self.labels3
        elif cat == 'all':
            if dataType == 'test':
                self.file_name = 'answers.txt'
                self.labels = lables_test
            else:
                raise ValueError('Invalid dataType')
            
        self.patch_size = patch_size

        self.vqas = []
        print(self.data_floder_loc, self.file_name)
        file_data = open((os.path.join(self.data_floder_loc, self.file_name)), "r")
        lines = [line.strip("\n") for line in file_data if line != "\n"]
        file_data.close()
        # print("lines:",lines)
        for line in lines: 
            if(self.cat!='all' and line.split('|')[1]!=self.cat_name_map[self.cat]):
                continue
            self.vqas.append([line])

        print('Total question: %.d' % len(self.vqas))

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
        answer = str(self.vqas[idx][0].split('|')[3])
        # answer=self.labels.index(answer)


        # print("1:/n", self.vqas[idx][0].split('|')[0], "2:/n", image_tensor, "3:/n", question, "4:/n", answer)

        return self.vqas[idx][0].split('|')[0], image_tensor, question, answer



def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print("model name:",model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type,local_files_only=True)

    # questions = pd.read_table(os.path.expanduser(args.question_file))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ans_file = open(output_file, "w")

    
    dataset = MedVQAClassification(datafloder=args.data_floder, imgfloder=args.image_floder,
                                   image_processor=image_processor, model=model, cat=args.category,
                                   dataType=args.dataType)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0

    options = dataset.getOptions()
    options_str=", ".join(options)
    print("Total options:{}, random select acc{:.4f}".format(len(options), 1 / len(options)))
    rightAnswer=[]
    prediction=[]
    rightAnswer_idx=[]

    for i,(idx_image, image_tensor, question, answer) in enumerate(tqdm(dataloader, total=len(dataset))):

        question = " ".join(question)
        # answer_inx=answer
        answer = " ".join(answer)

        hint = question + '\nYou have the following {} options to choose from:\n'.format(len(options)) + options_str

        # qs = hint + '\n' + question

        if args.single_pred_prompt:
            if args.lang == 'cn':
                qs = hint + '\n' + "请直接从待选的选项中选择一个回答，只回答选项内容，不要有多余输出。"
            else:
                qs = hint + '\n' + "Please choose an answer directly from the options provided, and only provide the content of the chosen option without any additional output."

            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=128,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            if('#' in answer): # 多择处理Acc，跳过prediction和rightAnswer列表添加
                answer_list=answer.split('#')
                if(outputs.lower() in answer_list):
                    count = count + 1
                    rightAnswer_idx.append(i+1)
            else: # 单选处理
                prediction.append(outputs.lower())
                rightAnswer.append(answer.lower())

                # ans_id = shortuuid.uuid()
                if (outputs.lower() == answer.lower()):
                    count = count + 1
                    rightAnswer_idx.append(i+1)
            ans_file.write(json.dumps({"question_id": idx_image,
                                       "prompt": qs,
                                       "outputs": outputs,
                                       "answer": answer,
                                       # "answer_id": ans_id,
                                       "model_id": model_name,
                                       "metadata": {}}) + "\n")
            ans_file.flush()

            # # rotate options
            # options = options[1:] + options[:1]
            # cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    
    print(f"正确回答的样本编号：:{rightAnswer_idx}")
    print("ACC:{}/{}   {:.4f}".format(count, len(dataset), count / len(dataset)))

    # Transform answers and predictions to numerical labels
    label_encoder = LabelEncoder()
    all_possible_answers=[]
    for item in options:
        all_possible_answers.append(item.lower())
    assert len(all_possible_answers) == len(set(all_possible_answers)), "There are duplicate answers in the options."
    label_encoder.fit(all_possible_answers)

    rightAnswer = label_encoder.transform(rightAnswer)
    prediction = label_encoder.transform(prediction)
    assert len(rightAnswer) == len(prediction), "The lengths of rightAnswer and prediction must be the same."


    # 计算Recall（macro平均）
    recall_macro = recall_score(rightAnswer, prediction, average='macro')
    print(f"Recall (Macro):{recall_macro:.4f}")

    # 计算Recall（micro平均）
    recall_micro = recall_score(rightAnswer, prediction, average='micro')
    print(f"Recall (Micro):{recall_micro:.4f}")

    # 计算F1 Score（macro平均）
    f1_macro = f1_score(rightAnswer, prediction, average='macro')
    print(f"F1 Score (Macro):{f1_macro:.4f}")

    # 计算F1 Score（micro平均）
    f1_micro = f1_score(rightAnswer, prediction, average='micro')
    print(f"F1 Score (Micro):{f1_micro:.4f}")
    
    summary = {
        "total_samples": len(dataset),
        "correct_predictions": count,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }
    ans_file.write(json.dumps(summary) + "\n")
    
    ans_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--data-floder", type=str, default=None)
    parser.add_argument("--image-floder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--dataType", type=str, default="test")

    args = parser.parse_args()

    eval_model(args)
