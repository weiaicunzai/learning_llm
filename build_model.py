from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image

def save_weights():

    clip_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/model_weights/clip_vit'
    qwen_dir = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/model_weights/qwen1.5-0.5b"

    qwen_model = AutoModelForCausalLM.from_pretrained(qwen_dir)
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_dir)

    clip_model = AutoModel.from_pretrained(clip_path)

    qwen_config = qwen_model.config
    vit_config = clip_model.vision_model.config

    llava_config = LlavaConfig(vit_config, qwen_config)
    llava_model = LlavaForConditionalGeneration(llava_config)
    llava_model.vision_tower.vision_model = clip_model.vision_model
    llava_model.language_model = qwen_model

    llava_model.config.pad_token_id = qwen_tokenizer.pad_token_id
    llava_model.config.image_token_index = qwen_tokenizer.encode('<image>')[0]


    # save 
    llava_model.save_pretrained('pretrained_model/model001')
    qwen_tokenizer.save_pretrained("pretrained_model/model001")
    autoprocessor = AutoProcessor.from_pretrained(clip_path)
    autoprocessor.save_pretrained('pretrained_model/model002')



# test 
def test():
    model_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/pretrained_model/model001'

    llava_processor = LlavaProcessor.from_pretrained(model_path)
    llava_tokenizer = AutoTokenizer.from_pretrained(model_path)
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_path)

    prompt_text = "<image>\nWhat are these?"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]

    prompt = llava_processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/000000039769.jpg"
    image = Image.open(image_path)

    inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

    print(inputs)
    # for tk in inputs.items():
        # print(tk)
    for tk in inputs.keys():
        inputs[tk] = inputs[tk].to(llava_model.device)

    generate_ids = llava_model.generate(**inputs, max_new_tokens=20)
    # print(generate_ids)

    gen_text = llava_processor.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    print(gen_text)




    # print(prompt)





if __name__ == '__main__':
    # save_weights()
    test()



