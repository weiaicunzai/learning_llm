from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import LlavaForConditionalGeneration, Trainer

from dataset import PretrainData, Collator




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/by/train_llava/pretrained_model/model001")
    train_type: Optional[str] = field(
        default="freeze_all",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            """
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/liuhaotian/LLaVA-Pretrain/main/', metadata={"help": "Path to the training data."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)
    
def load_model_processor(model_args: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
        )
    
    processor = transformers.AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if model_args.train_type == 'lora':
        print('loading model to Lora')

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )

        model = get_peft_model(model, config)

    elif model_args.train_type == "none":
        print("使用全量参数进行训练")

    elif model_args.train_type == 'freeze_vision':
        print('freeze vision tower')
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    elif model_args.train_type == 'freeze_all':
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        
        for param in model.language_model.parameters():
            param.requires_grad = False

    return model, processor


def load_dataset_collator(processor, data_args: DataArguments):
    llava_dataset = PretrainData(data_args.data_path, processor, -100)
    collator = Collator(processor.tokenizer.pad_token_id)

    return llava_dataset, collator




def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, transformers.TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_processor(model_args)
    print(model)

    dataset, collator = load_dataset_collator(processor, data_args)
    training_args.fp16 = True
    # print(training_args)
    training_args.dataloader_num_workers=8
    training_args.dataloader_pin_memory=True
    training_args.dataloader_persistent_workers=True
    training_args.per_gpu_train_batch_size = 100
    training_args.learning_rate = 4e-4
    training_args.gradient_accumulation_steps = 8
    training_args.num_train_epochs = 10

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        # num_train_epochs=3
    )
    print(training_args.output_dir)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)







train()
