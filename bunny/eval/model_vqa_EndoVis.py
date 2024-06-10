import argparse
import torch
import os
import glob
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

from bunny.util.data_utils import DataArguments




class EndoVis18VQAClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'alldata/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    	patch_size      = 1/2/3/4/5
    '''
    def __init__(self,data_floder: str,
                 dataType='test',
                 eval_model=None,
                 eval_image_processer=None):
             
        self.data_floder=data_floder
        self.floder_head = 'alldata/seq_'
        self.floder_tail = '/vqa/Classification/*.txt'

        if dataType == 'test':
            self.seq=[1]
            assert eval_model is not None, "eval_model is required for validation"
            assert eval_image_processer is not None, "eval_image_processer is required for validation"
            self.model=eval_model
            self.image_processor=eval_image_processer
        elif dataType == 'train':
            self.seq=[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        
        # files, question and answers
        filenames = []
        # str1=self.data_floder+'/'+self.floder_head + str(1) + self.floder_tail
        for curr_seq in self.seq: filenames = filenames + glob.glob(self.data_floder+'/'+self.floder_head + str(curr_seq) + self.floder_tail)
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
        image_path=os.path.join(self.data_floder,'alldata/'+seq,'left_frames',image_name+'.png')
        image = load_image_from_path(image_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        
        
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
        return image_name, image_tensor, question, answer


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

    
    dataset = EndoVis18VQAClassification(args.data_floder, args.dataType, model, image_processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0

    options = dataset.getOptions()
    options_str=", ".join(options)
    print("Total options:{}, random select acc{:.4f}".format(len(options), 1 / len(options)))
    rightAnswer=[]
    prediction=[]
    rightAnswer_idx=[]

    for i,(idx_image, image_tensor, question_list, answer_list) in enumerate(tqdm(dataloader, total=len(dataset))):
        for question,answer in zip(question_list,answer_list):
            question=''.join(question)
            answer=''.join(answer)

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
                        print("yeah,",count)
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
    print("ACC:{}/{}   {:.4f}".format(count, dataset.total_question, count / dataset.total_question))

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
