import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional

import torch

import transformers

from bunny.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from torch.utils.data import Dataset

from bunny import conversation as conversation_lib

from bunny.util.mm_utils import tokenizer_image_token

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
        question: str,
        answer:str,
        options: Sequence[str],
        data_args: DataArguments
)->Dict:
    question=DEFAULT_IMAGE_TOKEN+question
    if DEFAULT_IMAGE_TOKEN in question:
        question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
        question = question.strip()
    replace_token = DEFAULT_IMAGE_TOKEN

    question = question.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # 添加多项选择任务的提示
    options_str=', '.join(options)
    hint = question + '\nYou have the following {} options to choose from:\n'.format(len(options)) + options_str

    qs = hint + '\n' + "Please choose an answer directly from the options provided, and only provide the content of the chosen option without any additional output."

    ans=[]
    ans.append({"from": "human", "value": qs})
    ans.append({"from": "gpt", "value": answer})
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


        if data_args.category == 'cat1':
            if data_args.dataType == 'val':
                self.file_name = 'val/QAPairsByCategory/C1_Modality_val.txt'
            elif data_args.dataType == 'train':
                self.file_name = 'train/QAPairsByCategory/C1_Modality_train.txt'
            elif data_args.dataType == 'test':
                self.file_name = 'test/QAPairsByCategory/C1_Modality_test.txt'
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
        elif data_args.category == 'cat2':
            if data_args.dataType == 'val':
                self.file_name = 'val/QAPairsByCategory/C2_Plane_val.txt'
            elif data_args.dataType == 'train':
                self.file_name = 'train/QAPairsByCategory/C2_Plane_train.txt'
            elif data_args.dataType == 'test':
                self.file_name = 'test/QAPairsByCategory/C2_Plane_test.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = ['axial', 'longitudinal', 'coronal', 'lateral', 'ap', 'sagittal', 'mammo - mlo', 'pa',
                           'mammo - cc', 'transverse', 'mammo - mag cc', 'frontal', 'oblique', '3d reconstruction',
                           'decubitus', 'mammo - xcc']
        elif data_args.category == 'cat3':
            if data_args.dataType == 'val':
                self.file_name = 'val/QAPairsByCategory/C3_Organ_val.txt'
            elif data_args.dataType == 'train':
                self.file_name = 'train/QAPairsByCategory/C3_Organ_train.txt'
            elif data_args.dataType == 'test':
                self.file_name = 'test/QAPairsByCategory/C3_Organ_test.txt'
            else:
                raise ValueError('Invalid dataType')
            self.labels = ['lung, mediastinum, pleura', 'skull and contents', 'genitourinary', 'spine and contents',
                           'musculoskeletal', 'heart and great vessels', 'vascular and lymphatic', 'gastrointestinal',
                           'face, sinuses, and neck', 'breast']
        elif data_args.category == 'all':
            if data_args.dataType == 'test':
                self.file_name = 'test/answers.txt'
            elif data_args.dataType=='train':
                self.file_name='train/all_qa_pairs_train.txt'
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
        train_dataset = MedVQAClassification(data_path=data_args.data_path,
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
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
