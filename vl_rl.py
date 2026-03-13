import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline
import json, pickle, random, re
import numpy as np
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from peft import PeftModel
import clip
from trl import GRPOConfig, GRPOTrainer
from easydict import EasyDict
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import argparse

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="KEVIN04087/SFT")
parser.add_argument("--output_dir", type=str, default="./")
args = parser.parse_args()

output_dir = args.output_dir
config = EasyDict()
config.toxic_coef = 5.0
config.keep_incentive = 0.3
config.num_images_per_prompt = 10
config.guidance_scale = 7.0



pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    use_safetensors=True,
    device_map="balanced",
    variant="fp16",
    safety_checker=None
)



model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)


model = PeftModel.from_pretrained(
    model,
    args.model_path,
    is_trainable=True,         
)

with open('ppo_data.json', 'r') as f:
    ppo_data = json.load(f)
formatted_data = [{'prompt': item['input']} for item in ppo_data]
dataset = Dataset.from_list(formatted_data)

chosen_labels = [
    "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "ARMPITS_EXPOSED",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED",
]

def load_prompts(device):
    file_path = 'prompts.p'
    return torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)

class Q16_score_optimized():
    def __init__(self):
        self.torch_device = "cuda"
        self.safety_prompts = load_prompts(device=self.torch_device)
        self.safety_prompts = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.torch_device)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
    @torch.no_grad()
    def score(self, images: Image):
        clip_inputs = self.processor(images, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_inputs.pixel_values).image_embeds
        safety_prompts_norm = self.safety_prompts
        image_features_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return (100 * image_features_norm.unsqueeze(1) * safety_prompts_norm.unsqueeze(0)).sum(dim=-1).softmax(dim=-1)[:, 1]


optimized_q16 = Q16_score_optimized()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
clip_model, preprocess = clip.load('ViT-B/32', device="cuda")

def get_clip_score_optimized(images, no_keep_prompts):
    
    truncated_prompts = no_keep_prompts

    image_inputs = torch.concat([preprocess(img).unsqueeze(0).to("cuda") for img in images], dim=0)
    text_inputs = clip.tokenize(truncated_prompts, truncate=True).to("cuda").repeat_interleave(config.num_images_per_prompt, dim=0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_inputs)
        text_features = clip_model.encode_text(text_inputs)
        
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_scores = (image_features * text_features).sum(dim=-1)
    return clip_scores

def generate_images(sd_pipeline, prompts: list[str]):
    
    if not prompts:
        return []
    return sd_pipeline(
        prompts,
        guidance_scale=config.guidance_scale,
        num_images_per_prompt=config.num_images_per_prompt
    ).images


def optimized_reward_func(completions, prompts):
    is_keep = [c.strip().lower() == 'keep' for c in completions]
    no_keep_completions = [completions[i] for i, k in enumerate(is_keep) if not k]
    no_keep_prompts     = [prompts[i]     for i, k in enumerate(is_keep) if not k]

    
    
    if not no_keep_completions:
        info = EasyDict()
        info.generated_images = [None] * len(completions)
        info.total_rewards    = [config.keep_incentive] * len(completions)
        info.clip_rewards     = [None] * len(completions)
        info.toxic_rewards    = [None] * len(completions)
        info.extract_prompts  = completions
        return info

    
    generated_images = generate_images(pipe, no_keep_completions)
    clip_scores = (
        get_clip_score_optimized(generated_images, no_keep_prompts)
        .reshape(len(no_keep_completions), -1)
        .mean(axis=-1)
    )
    toxic_scores = (
        optimized_q16.score(generated_images)
        .reshape(len(no_keep_completions), -1)
        .mean(axis=-1)
    )

    total_score = clip_scores + (-config.toxic_coef * toxic_scores + config.toxic_coef)

    results = []
    count = 0
    for keep in is_keep:
        if keep:
            
            results.append((None, config.keep_incentive, None, None))
        else:
            
            idx = random.randint(0, config.num_images_per_prompt - 1)
            img = generated_images[count * config.num_images_per_prompt + idx]
            results.append((
                img,
                total_score[count].item(),
                clip_scores[count].item(),
                toxic_scores[count].item()
            ))
            count += 1

    generated_images, total_rewards, clip_rewards, toxic_rewards = map(list, zip(*results))
    info = EasyDict()
    info.generated_images = generated_images
    info.total_rewards    = total_rewards
    info.clip_rewards     = clip_rewards
    info.toxic_rewards    = toxic_rewards
    info.extract_prompts  = completions
    return info



output_dir = args.output_dir
run_name = "VL-RAPID"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=0.000019,
    lr_scheduler_type='constant',
    save_strategy="steps",
    logging_steps=5,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=5,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=0.3,
    report_to="wandb",
    log_completions=True,
    max_steps=100,
    vllm_device="cuda:3",
    vllm_max_model_len=int(16400384),
    beta=0.0001,
    vllm_dtype="bfloat16",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[optimized_reward_func],
    args=training_args,
    train_dataset=dataset,
    pipe=pipe,
    processor=processor
)

trainer.train()
