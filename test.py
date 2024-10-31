import os
from PIL import Image

import pandas as pd
from transformers import  LlavaForConditionalGeneration, AutoProcessor



def load_modal_and_processor(model_path):
    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def build_model_input(data_path, processor):
    # from dataset import PretrainData, Collator

    # return PretrainData(data_path, processor, -100), Collator(processor.tokenizer.pad_token_id)

    json_path = os.path.join(data_path, 'blip_laion_cc_sbu_558k.json')
    df = pd.read_json(json_path)
    name, image_path, conversations = df.iloc[55]
    image_path = os.path.join(data_path, 'images', image_path)
    human_input = conversations[0].get('value')
    chatbot_output = conversations[1].get('value')
    print('human:', human_input)
    print('chatbot:', chatbot_output)
    print('image_path', image_path)
    print(json_path)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": human_input},
    ]

    image = Image.open(image_path)
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt = processor(text=prompt, images=image, return_tensors='pt')



    return prompt







model, processor = load_modal_and_processor('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/pretrained_model/model001')
prompt = build_model_input('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/liuhaotian/LLaVA-Pretrain/main/', processor)

model.eval()

model = model.to('cuda:1')


for tk in prompt.keys():
    prompt[tk] = prompt[tk].to(model.device)

generate_ids = model.generate(**prompt, max_new_tokens=100)


generate_ids = [
    oid[len(iids):] for oid, iids in zip(generate_ids, prompt.input_ids)
]

gen_text = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

print('pred:', gen_text)

import sys; sys.exit()

# model.eval()

# model.to('cuda:0')

# sample = collator([dataset[33], dataset[44], dataset[66]])

# for key in sample.keys():
#     sample[key] = sample[key].to(model.device)

# out = [
#         oid[:(iids == -100).sum()] for oid, iids in zip(sample[], sample['labels'])
#     ]

# out = model.generate(**sample, max_new_tokens=50)

# out = [
#         oid[:(iids == -100).sum()] for oid, iids in zip(out, sample['labels'])
#     ]




# gen_text = processor.batch_decode(out, skip_special_tokens=False, clean_up_tokenization_spaces=False)

# for text in gen_text:
#     print(text)

