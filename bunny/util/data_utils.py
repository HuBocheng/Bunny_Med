import os
import glob
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional

import torch

import transformers

from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset,ConcatDataset

from bunny import conversation as conversation_lib

from bunny.util.mm_utils import tokenizer_image_token, process_images,load_image_from_path, \
    get_model_name_from_path

from PIL import Image

from bunny.util.surgical_datasets import *


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    data_floder: str = field(default=None, metadata={"help": "Path to the training data floder."})
    dataType: str = field(default=None, metadata={"help": "Type of Med_VQA19 data (train/val/test)."})
    category: str = field(default=None, metadata={"help": "Category of Med_VQA19 data (cat1/cat2/cat3/all)."})



def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
    
            replace_token = DEFAULT_IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    return sources

def preprocess_MedVQA(
        question: Sequence[str],
        answer:Sequence[str],
        options: Sequence[str],
        data_args: DataArguments
)->Dict:
    options_str=', '.join(options)
    question[0]=DEFAULT_IMAGE_TOKEN+question[0]

    ans=[]
    for i in range(len(question)):
        q,a=question[i],answer[i]
        hint = q + '\nYou have the following {} options to choose from:\n'.format(len(options)) + options_str
        q=hint + '\n' + "Please choose an answer directly from the options provided, and only provide the content of the chosen option without any additional output."
        ans.append({"from": "human", "value": q})
        ans.append({"from": "gpt", "value": a})
    
    # 添加多项选择任务的提示
    # options_str=', '.join(options)
    # hint = question + '\nYou have the following {} options to choose from:\n'.format(len(options)) + options_str
    
    # qs = hint + '\n' + "Please choose an answer directly from the options provided, and only provide the content of the chosen option without any additional output."
    return [ans]


def preprocess_bunny(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
    
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations
    
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
    
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        end_token_cnt = 0
    
        for i, rou in enumerate(rounds):
            if rou == "":
                break
    
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
    
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
    
            round_len += 1
            end_token_cnt += 1
    
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
    
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
    
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_bunny_with_bos(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
    
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations
    
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    targets = input_ids.clone()
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
    
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        end_token_cnt = 0
        target[:cur_len] = IGNORE_INDEX
    
        for i, rou in enumerate(rounds):
            if rou == "":
                break
    
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
    
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
    
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
    
            end_token_cnt += 1
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
    
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)

    if conversation_lib.default_conversation.version == "bunny": # 用的是bunny的聊天模板，进入preprocess_bunny这个函数
        return preprocess_bunny(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version in {"minicpm", "llama"}:
        return preprocess_bunny_with_bos(sources, tokenizer, has_image=has_image)



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
    
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(MedVQAClassification, self).__init__()


        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_floder_loc=data_args.data_floder
    
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
                        'heart and great vessels#lung, mediastinum, pleura', 'lung, mediastinum, pleura#spine and contents', 'heart and great vessels#lung, mediastinum, pleura#spine and contents', 'ct with gi and iv contrast', 'longitudinal', 'skull and contents#spine and contents', '3d reconstruction', 'frontal', 'noncontrast', 'genitourinary', 'xr - plain film', 'flair', 'cta - ct angiography', 'ct - gi & iv contrast', 'an - angiogram']
    
        if data_args.category == 'cat1':
            if data_args.dataType == 'val':
                self.file_name = 'val/QAPairsByCategory/C1_Modality_val.txt'
            elif data_args.dataType == 'train':
                self.file_name = 'train/QAPairsByCategory/C1_Modality_train.txt'
            elif data_args.dataType == 'test':
                self.file_name = 'test/QAPairsByCategory/C1_Modality_test.txt'
            else:
                raise ValueError('Invalid dataType')  # 抛出错误
            self.labels = self.labels1
        
        elif data_args.category == 'cat2':
            if data_args.dataType == 'val':
                self.file_name = 'val/QAPairsByCategory/C2_Plane_val.txt'
            elif data_args.dataType == 'train':
                self.file_name = 'train/QAPairsByCategory/C2_Plane_train.txt'
            elif data_args.dataType == 'test':
                self.file_name = 'test/QAPairsByCategory/C2_Plane_test.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = self.labels2
        elif data_args.category == 'cat3':
            if data_args.dataType == 'val':
                self.file_name = 'val/QAPairsByCategory/C3_Organ_val.txt'
            elif data_args.dataType == 'train':
                self.file_name = 'train/QAPairsByCategory/C3_Organ_train.txt'
            elif data_args.dataType == 'test':
                self.file_name = 'test/QAPairsByCategory/C3_Organ_test.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = self.labels3
        elif data_args.category == 'all':
            if data_args.dataType == 'test':
                self.file_name = 'test/answers.txt'
            elif data_args.dataType=='train':
                self.file_name='train/WithoutC4.txt'
                self.labels = self.labels1+self.labels2+self.labels3
            else:
                raise ValueError('Invalid dataType')
            
        self.vqas = []
        print(self.data_floder_loc, self.file_name)
        file_data = open((os.path.join(self.data_args.data_floder, self.file_name)), "r")
        lines = [line.strip("\n") for line in file_data if line != "\n"]
        file_data.close()
        # print("lines:",lines)
        for line in lines: self.vqas.append([line])
        print('Total question: %.d' % len(lines))
  

    def __len__(self):
        return len(self.vqas)
    
    def __getitem__(self, idx):
        # img
        processor = self.data_args.image_processor
        self.image_path = os.path.join(self.data_args.data_floder,self.data_args.image_folder ,self.vqas[idx][0].split('|')[0]) + '.jpg'
        image = load_image_from_path(self.image_path)
    
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
    
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # question and answer
        question = self.vqas[idx][0].split('|')[1]
        answer = self.vqas[idx][0].split('|')[2]
    
        # print("1:/n", self.vqas[idx][0].split('|')[0], "2:/n", image_tensor, "3:/n", question, "4:/n", answer)
        sources=preprocess_MedVQA([question], [answer], self.getOptions(), self.data_args)
        # sources = preprocess_multimodal(
        #     copy.deepcopy([e["conversations"] for e in sources]), self.data_args)


        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict['image'] = image
        if self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict



class EndoVis18VQAClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'alldata/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    	patch_size      = 1/2/3/4/5
    '''
    def __init__(self,data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 eval_model=None,
                 eval_image_processer=None):
             
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_floder_loc=data_args.data_floder
        self.floder_head = 'alldata/seq_'
        self.floder_tail = '/vqa/Classification/*.txt'
    
        if data_args.dataType == 'val':
            self.seq=[1, 5, 16]
            assert eval_model is not None, "eval_model is required for validation"
            assert eval_image_processer is not None, "eval_image_processer is required for validation"
            self.model=eval_model
            self.image_processor=eval_image_processer
        elif data_args.dataType == 'train':
            self.seq=[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        
        # files, question and answers
        filenames = []
        # str1=self.data_floder_loc+'/'+self.floder_head + str(1) + self.floder_tail
        for curr_seq in self.seq: filenames = filenames + glob.glob(self.data_floder_loc+'/'+self.floder_head + str(curr_seq) + self.floder_tail)
        self.vqas = []
        self.total_question=0
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            self.vqas.append([file, lines])
            self.total_question+=len(lines)
        print('Total files: %d | Total question: %.d' %(len(filenames), self.total_question))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom']
        
    def __len__(self):
        return len(self.vqas)
    
    def getOptions(self):
        return self.labels
    
    def __getitem__(self, idx):
        
        # img
        image_name = self.vqas[idx][0].split('/')[-1].split('_')[0]
        seq=self.vqas[idx][0].split('/')[-4]
        image_path=os.path.join(self.data_args.data_floder,'alldata/'+seq,'left_frames',image_name+'.png')
        image = load_image_from_path(image_path)
    
        if self.data_args.dataType == 'val':
            processor=self.image_processor
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


        else:
            processor = self.data_args.image_processor
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # frame_data = h5py.File(visual_feature_loc, 'r')    
        # visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        # question = self.vqas[idx][1].split('|')[0]
        # label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
    
        # return loc[-1].split('_')[0], visual_features, question, label
        # question and answer
        QAs=self.vqas[idx][1]
        question=[]
        answer=[]
        for item in QAs:
            question.append(item.split('|')[0])
            answer.append(item.split('|')[1])
        # print("1:/n", self.vqas[idx][0].split('|')[0], "2:/n", image_tensor, "3:/n", question, "4:/n", answer)
        sources=preprocess_MedVQA(question, answer, self.getOptions(), self.data_args)
        # sources = preprocess_multimodal(
        #     copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
    
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict['image'] = image
        if self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
    
        list_data_dict = json.load(open(data_path, "r"))
    
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
    
    def __len__(self):
        return len(self.list_data_dict)
    
    @property
    def lengths(self): # 计算数据集中每个样本的长度并返回长度列表
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            # 将每个对话的文本内容分割成单词，然后计算单词的数量来得到对话的长度
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list
    
    @property
    def modality_lengths(self):
        # 如果样本包含图像，那么长度为对话的长度；如果样本不包含图像，那么长度为对话的长度的负值。
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
    
                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
    
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
    
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
    
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
    
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
    
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
    
        labels = labels[:, :self.tokenizer.model_max_length]
    
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
    
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
    
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    if(data_args.category is not None):
        print("MedVQA is selected.")
        train_dataset = MedVQAClassification(data_path=data_args.data_floder,
                                             tokenizer=tokenizer,
                                             data_args=data_args)
    else:
        if 'EndoVis' in data_args.data_floder:
            print("EndoVis18 is selected.")
            train_dataset = EndoVis18VQAClassification(data_path=data_args.data_floder,
                                                        tokenizer=tokenizer,
                                                        data_args=data_args)
        else:
            train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_surgical_data_module(tokenizer: transformers.PreTrainedTokenizer,
                              data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_datasets=[]
   
    train_dataset1 = MedVQAClassification(data_path=data_args.data_floder,
                                             tokenizer=tokenizer,
                                             data_args=data_args)
    all_datasets.append(train_dataset1)
    
    
    train_dataset2 = EndoVis18VQAClassification(data_path=data_args.data_floder,
                                                    tokenizer=tokenizer,
                                                    data_args=data_args)
    all_datasets.append(train_dataset2)

    train_dataset=ConcatDataset(all_datasets)
        
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
