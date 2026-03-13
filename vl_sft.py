from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--dataset_dir", type=str, default="./")
parser.add_argument("--output_dir", type=str, default="./")
args = parser.parse_args()
def format_data(sample):
     return {"messages" : [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """
        You are an AI safety moderation model. Your task is to check and refine prompts intended for a text-to-image generation model, ensuring they comply strictly with AI safety guidelines and ethics.
        You will be given both an image and its corresponding text prompt. Use **both** the visual content and the text prompt to make your judgment.
        - If the prompt or the image contains any unsafe, inappropriate, harmful, or unethical content, **rewrite the prompt** to make it safe while preserving the original intent as closely as possible.
        - If the image already safe and ethical, simply respond with: keep
        """
                    },

                    
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"]
                    },
                    {
                        "type": "text",
                        "text": f"Original Prompt: {sample['query']}"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["response"]
                    }
                ]
            }
        ]
    }


dataset = load_dataset("KEVIN04087/ToxiClean-IT", split = "train") 
 


dataset = [format_data(sample) for sample in dataset]
 
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
 

model_id = args.model_id
 

 

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(model_id)

from peft import LoraConfig
 

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", 
)

from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
 
args = SFTConfig(
    output_dir=args.output_dir, 
    num_train_epochs=3,                     
    per_device_train_batch_size=4,          
    gradient_accumulation_steps=4,          
    gradient_checkpointing=True,            
    logging_steps=10,                       
    save_strategy="epoch",                  
    learning_rate=5e-5,     
    max_steps=636,                          
    bf16=True,                              
    tf32=True,                              
    lr_scheduler_type="constant",           
    report_to="wandb",                
    gradient_checkpointing_kwargs = {"use_reentrant": False}, 
    dataset_text_field="", 
    dataset_kwargs = {"skip_prepare_dataset": True}, 
    push_to_hub=True,                       

)
args.remove_unused_columns=False
 

def collate_fn(examples):
    
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
 
    
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
 
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  
    
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
 
    return batch

from trl import SFTTrainer
 
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
)

trainer.train()