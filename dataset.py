import os
from transformers import AutoProcessor, AutoTokenizer
import pandas as pd
from dataclasses import dataclass
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


from dataclasses import dataclass

@dataclass
class Output:
    def __init__(
        self,
        question_input_ids = None,
        answer_input_ids = None,
        pixel_values = None,
    ):
        self.question_input_ids = question_input_ids
        self.answer_input_ids = answer_input_ids
        self.pixel_values = pixel_values

class PretrainData(Dataset):
    def __init__(self, path, processor, ignore_idx):
        self.path = path

        json_path = os.path.join(path, 'blip_laion_cc_sbu_558k.json')
        self.data = pd.read_json(json_path)
        self.image_dir = os.path.join(path, 'images')
        self.processor = processor
        self.ignore_idx = ignore_idx

    def build_prompt(self, text, image_path, answer):

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors='pt', padding='longest')

        # print('answer:', answer)
        answer  = self.processor.tokenizer(answer, return_tensors='pt', padding='longest')
        answer = answer['input_ids']

        return Output(
            question_input_ids=inputs['input_ids'],
            answer_input_ids=answer,
            pixel_values=inputs['pixel_values']
        )



    def __getitem__(self, index):
        name, image_path, conversations = self.data.iloc[index]
        image_path = os.path.join(self.image_dir, image_path)
        human_input = conversations[0].get('value')
        chatbot_output = conversations[1].get('value')
        outputs = self.build_prompt(human_input, image_path, chatbot_output)

        input_ids = torch.cat([
            outputs.question_input_ids, 
            outputs.answer_input_ids
        ], dim=1)

        # # print(outputs.question_input_ids.shape)
        # # print(torch.tensor(outputs.question_input_ids.shape, dtype=outputs.question_input_ids.dtype).shape)
        labels = torch.cat([
            outputs.question_input_ids * 0  + self.ignore_idx,
            outputs.answer_input_ids
        ], dim=1)
        # print(labels)
        # print(labels.shape)

        # return {
        #     'input_ids': input_ids[0],
        #     'label': labels[0],
        #     'pixel_values': outputs.pixel_values[0]
        # }
        # return {
        #     'input_ids': input_ids,
        #     'labels': labels,
        #     'pixel_values': outputs.pixel_values
        # }
        # print(11111111, input_ids)
        return [
            input_ids,
            labels,
            outputs.pixel_values
        ]

    def __len__(self):
        return self.data.shape[0]

class Collator:
    def __init__(self, pad_value) -> None:
        self.pad_value = pad_value

    def __call__(self, features):
        labels_list = []
        input_ids = []
        pixel_values = []
        max_input_len_list = []
        for feat in features:
            labels = feat[1]
            pixel_values.append(feat[2])
            max_input_len_list.append(feat[0].shape[1])
            labels_list.append(labels)
            input_ids.append(feat[0])


        max_input_len = max(max_input_len_list)

        final_input_ids =  [
            torch.cat(
                [torch.full((1, max_input_len - ids.shape[1]), self.pad_value), ids], axis=1
            ) 
            for ids in input_ids
        ]
        final_labels = [
            torch.cat(
                [
                    torch.full((1, max_input_len - ids.shape[1]), self.pad_value), ids
                ],
                axis=1
            )
            for ids in labels_list
        ]
       
        final_input_ids = torch.cat(final_input_ids, axis=0)
        final_labels = torch.cat(final_labels, axis=0)
        final_pixel_values = torch.cat(pixel_values, axis=0)

        attention_mask = torch.ones_like(
            final_input_ids
        )

        attention_mask[final_input_ids == self.pad_value] = 0

        return {
            'input_ids': final_input_ids,
            'labels': final_labels,
            'pixel_values': final_pixel_values,
            'attention_mask': attention_mask
        }






# path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/liuhaotian/LLaVA-Pretrain/main/'

# pretrained_weights = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/pretrained_model/model001'

# processor = AutoProcessor.from_pretrained(pretrained_weights)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
# dataset = PretrainData(path, processor, ignore_idx=-100)
# #collator = DataCollatorWithPadding(tokenizer, padding=True)
# collator = Collator(pad_value=tokenizer.pad_token_id)

# print(processor.tokenizer.eos_token_id)
# print(processor.tokenizer.pad_token_id)

# ffs = []
# # for item in dataset:
#     # print(item)
# out = collator([dataset[3], dataset[4235], dataset[444], dataset[5555]])
# print(out)
# print(dataset[33]['pixel_values'].shape)
# print(dataset[44]['input_ids'].shape)
# print(dataset[44]['label'].shape)
# import sys; sys.exit()
# print(dataset[33]['label'].shape, dataset[44]['label'].shape)
#out = collator([dataset[33], dataset[44], dataset[55]])
# examples = [
#     {"input_ids": [1, 2, 3], "label": [0]},
#     {"input_ids": [4, 5], "label": [1, 1, 1]},
# ]
# out = collator(examples)
# print(out)
# print(out)
    # print(item)
# # print(len(dataset))
# for i in dataset:
#     print(i[0])
#     print(i[1])
#     print(i[2])

