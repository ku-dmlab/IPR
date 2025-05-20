# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import functools
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
import torch
import torch.utils.data
import transformers
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
import torchvision.transforms as T
from PIL import Image
from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..extras.profiling import profiling_context, profiling_decorator
from ..import_utils import is_rich_available, is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from transformers import Qwen2_5_VLForConditionalGeneration
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


import hashlib
import base64
import os
import random
from PIL import Image
from qwen_vl_utils import process_vision_info


import numpy as np
import clip
import pickle
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
clip_model, preprocess = clip.load('ViT-B/32',device="cuda")

def load_prompts(device):
    file_path = 'prompts.p'
    return torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)

class Q16_score():
    def __init__(self):
        self.torch_device = "cuda"
        self.safety_prompts = load_prompts(device=self.torch_device)

        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(self.torch_device)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def q16_classifier(self, embeddings, verbose=True):
        safety_prompts_norm = self.safety_prompts / self.safety_prompts.norm(dim=-1, keepdim=True)
        image_features_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ safety_prompts_norm.T)
        
        probs = similarity.squeeze().softmax(dim=-1)
        if verbose:
            print(probs.float())
        return probs[1]

    @torch.no_grad()
    def score(self, image: Image):
        clip_input = self.processor(images=image, return_tensors="pt").to(self.torch_device)
        image_embeds = self.model(clip_input.pixel_values).image_embeds
        score = self.q16_classifier(image_embeds,False)
        return score.item()
m = Q16_score()
def generate_image(sd_pipeline, prompt: str):
        result = sd_pipeline(prompt, guidance_scale = 7)
        return result.images[0]
    
def make_input(prompts, image_manager, pipe, processor, device):
    messages = []
    image_path = []
    for prompt in prompts:
        image_file_path, index = image_manager.sample(prompt, pipe)
        image_path.append(image_file_path)
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
        You are an AI safety moderation model. Your task is to check and refine prompts intended for a text-to-image generation model, ensuring they comply strictly with AI safety guidelines and ethics.
        You will be given both an image and its corresponding text prompt. Use **both** the visual content and the text prompt to make your judgment.
        - If the prompt or the image contains any unsafe, inappropriate, harmful, or unethical content, **rewrite the prompt** to make it safe while preserving the original intent as closely as possible.
        - If the image already safe and ethical, simply respond with: keep
        """
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file_path
                    },
                    {
                        "type": "text",
                        "text": f"Original Prompt: {prompt}"
                    }
                ]
            }
        ]
        messages.append(message)
    texts = [processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs.to(device), texts, image_path
class ImageManager:
    def __init__(self, base_path='./images_0.2'):
        """Initialize the ImageManager with a base path for storing images."""
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
    
    def _prompt_to_hash(self, prompt: str):
        """Convert a prompt to a hash value for directory naming."""
        hash_bytes = hashlib.sha256(prompt.encode()).digest()
        b64_str = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
        safe_str = b64_str.replace('=', '')
        return safe_str
    
    def _get_prompt_dir(self, prompt):
        """Get the directory path for a specific prompt."""
        hash_value = self._prompt_to_hash(prompt)
        return os.path.join(self.base_path, hash_value)
    
    def _get_image_files(self, prompt):
        """Get all image files for a specific prompt."""
        prompt_dir = self._get_prompt_dir(prompt)
        if not os.path.exists(prompt_dir):
            return []
        return [f for f in os.listdir(prompt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def add(self, user_prompt, prompt, sd_pipeline, index=None):
        """
        Add a new image for the given prompt.
        
        Args:
            prompt: The text prompt to generate the image
            sd_pipeline: The stable diffusion pipeline to use
            index: Optional specific index to use for the image
            
        Returns:
            tuple: (generated image, index used)
        """
        prompt_dir = self._get_prompt_dir(user_prompt)
        os.makedirs(prompt_dir, exist_ok=True)
        
        
        image = generate_image(sd_pipeline, prompt)
        
        
        if index is None:
            
            existing_files = self._get_image_files(user_prompt)
            indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('image_')]
            index = max(indices) + 1 if indices else 0
        
        
        image_file_path = os.path.join(prompt_dir, f"image_{index}.png")
        image.save(image_file_path)
        
        return image, index
    
    def sample(self, prompt, sd_pipeline=None, index=None):
        """
        Sample an image for the given prompt. Generate if none exists.
        
        Args:
            prompt: The text prompt
            sd_pipeline: Optional pipeline to generate image if none exists
            index: Optional specific index to retrieve
            
        Returns:
            tuple: (image, index)
        """
        image_files = self._get_image_files(prompt)
        
        if not image_files:
            
            if sd_pipeline is None:
                raise ValueError("No images found for this prompt and no pipeline provided to generate one")
            return self.add(prompt, prompt, sd_pipeline)
        
        if index is not None:
            
            target_file = f"image_{index}.png"
            if target_file in image_files:
                image_file_path = os.path.join(self._get_prompt_dir(prompt), target_file)
                return image_file_path, index
            else:
                raise ValueError(f"No image with index {index} found for this prompt")
        
        
        random_image_file = random.choice(image_files)
        image_file_path = os.path.join(self._get_prompt_dir(prompt), random_image_file)
        selected_index = int(random_image_file.split('_')[1].split('.')[0])
        
        return image_file_path, selected_index
    
    def remove(self, prompt, index):
        """
        Remove an image with the specified index for the given prompt.
        
        Args:
            prompt: The text prompt
            index: The index of the image to remove
            
        Returns:
            bool: True if successfully removed, False otherwise
        """
        prompt_dir = self._get_prompt_dir(prompt)
        image_file_path = os.path.join(prompt_dir, f"image_{index}.png")
        
        if os.path.exists(image_file_path):
            os.remove(image_file_path)
            return True
        return False
    
    def list(self, prompt):
        """
        List all available image indices for the given prompt.
        
        Args:
            prompt: The text prompt
            
        Returns:
            list: Sorted list of available image indices
        """
        image_files = self._get_image_files(prompt)
        indices = [int(f.split('_')[1].split('.')[0]) for f in image_files if f.startswith('image_')]
        return sorted(indices)
    
    def get_hash(self, prompt):
        """
        Get the hash value for a prompt.
        
        Args:
            prompt: The text prompt
            
        Returns:
            str: The hash value used for the directory name
        """
        return self._prompt_to_hash(prompt)






RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  
        
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        
        
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        
        
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats
            - [Conversational](dataset_formats
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        pipe = None,
        processor = None,
    ):
        self.image_manager = ImageManager()
        self.pipe = pipe
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        self.processor = processor
        
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        
        
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  
            elif isinstance(torch_dtype, str):  
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        
        self.beta = args.beta
        if self.beta == 0.0:
            
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto"
)
        elif is_peft_model(model):
            
            
            self.ref_model = None
        else:
            
            self.ref_model = create_reference_model(model)

        
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                
                
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        
        def data_collator(features):  
            return features

        
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  
        self.num_generations = args.num_generations  
        self.use_vllm = args.use_vllm
        self.temperature = args.temperature
        
        self.num_iterations = args.num_iterations  
        self.epsilon = args.epsilon
        
        self._step = 0
        
        
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        
        
        
        
        
        
        model.warnings_issued["estimate_tokens"] = True

        
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        
        
        
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                device_type = PartialState().default_device.type
                device_module = getattr(torch, device_type)
                if vllm_device == "auto":
                    if device_module.device_count() == 1:
                        vllm_device = f"{device_type}:0"  
                    else:
                        vllm_device = f"{device_type}:{self.accelerator.num_processes}"  
                
                if (
                    vllm_device.split(":")[0] == f"{device_type}"
                    and int(vllm_device.split(":")[1]) >= device_module.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {device_module.device_count() - 1}`."
                    )
                
                if vllm_device in {f"{device_type}:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                
                
                
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )

                
                
                
                
                
                
                @contextlib.contextmanager
                def new_group_context():
                    new_group = torch.distributed.new_group
                    try:
                        torch.distributed.new_group = functools.partial(new_group, use_local_synchronization=True)
                        torch.npu.mem_get_info = functools.partial(torch.npu.mem_get_info, device=vllm_device)
                        yield
                    finally:
                        torch.distributed.new_group = new_group

                new_group_patch = new_group_context() if device_type == "npu" else contextlib.nullcontext()
                with world_size_patch, profiling_patch, new_group_patch:
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                        self.llm = LLM(
                        model=model.name_or_path,
                        max_model_len=1024,
                        max_num_seqs=1,
                        mm_processor_kwargs={
                            "min_pixels": 28 * 28,
                            "max_pixels": 1280 * 28 * 28,
                            "fps": 1,
                        },
                        device=vllm_device,
                        
                        )

                
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None

                
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                )

            self._last_loaded_step = 0  

            
            
            
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        
        
        
        self.model_accepts_loss_kwargs = False

        
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        
        
        
        
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        
        model.config.use_cache = False

        
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, pixel_values, image_grid_thws):
        
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thws).logits
        logits = logits[:, :-1, :]  

        input_ids = input_ids[:, -logits_to_keep:]
        
        
        logits = logits[:, -logits_to_keep:]
        logits = logits/self.temperature
        return selective_log_softmax(logits, input_ids)  

    @profiling_decorator
    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            
            
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            
            inputs = self._generate_and_score_completions(inputs)
        return inputs


    def get_clip_score(self, image, prompt):

        
        

        image_input = preprocess(image).unsqueeze(0)
        text_input = clip.tokenize([prompt], truncate=True)
        
        image_input = image_input.to("cuda")
        text_input = text_input.to("cuda")
        
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
        
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        clip_score = torch.matmul(image_features, text_features.T).item()
        return clip_score
    
    def _compute_original_image_scores(self, image_paths: list, prompts: list[str]) -> list[float]:
        scores = []
        for path, prompt in zip(image_paths, prompts):
            if isinstance(path, str):
                
                pil_img = Image.open(path).convert("RGB")
            elif hasattr(path, "convert"):  
                pil_img = path
            else:
                raise ValueError("Unexpected type for image path: {}".format(type(path)))
            
            
            
            
            

            score_val = self.get_clip_score(pil_img, prompt)
            
            score_toxic = m.score(pil_img)
            scores.append(float(score_val)+float(-5*score_toxic+5))
        return scores

    def extract_xml_prompt(self,text: str) -> str:
        prompt = text.split("<prompt>")[-1]
        prompt = prompt.split("</prompt>")[0]
        return prompt.strip()
    
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [input['prompt'] for input in inputs]
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        inputs_ = make_input(prompts, self.image_manager, self.pipe, self.processor, device)
        image_paths = inputs_[2]

        original_scores = self._compute_original_image_scores(image_paths, prompts)
        original_scores = torch.tensor(original_scores, dtype=torch.float32, device=device ).unsqueeze(-1)
        
        
        
        
        
        
        
        prompt_ids = inputs_[0]['input_ids']
        prompt_mask = inputs_[0]['attention_mask']
        pixel_values = inputs_[0]['pixel_values']
        image_grid_thws = inputs_[0]['image_grid_thw']
        
        
        prompts_text = inputs_[1]
        
        
        
        

        
        if self.args.use_vllm:
            
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                
                
                
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            
            
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                      **{
                         "pixel_values": pixel_values,
                         "image_grid_thw": image_grid_thws,
                         "input_ids": prompt_ids,
                         "attention_mask": prompt_mask,
                    
                     },
                     max_new_tokens = 1000,
                     temperature = self.temperature
                )

            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            
                        
                        
                        
            
            
            
            
            
            
            
            
            
            


        
            

        
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  

        logits_to_keep = completion_ids.size(1)  

        with torch.no_grad():
            
            
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, pixel_values, image_grid_thws
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, pixel_values, image_grid_thws
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, pixel_values, image_grid_thws
                    )
        
        import re
        
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  
                else:
                    
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

                    
                    info = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    output_reward_func = info.total_rewards

                    
                    group_size = self.num_generations  
                    num_groups_local = len(prompts) // group_size

                    for g in range(num_groups_local):
                        start = g * group_size
                        end = start + group_size
                        group_generated = info.generated_images[start:end]
                        group_prompts = info.extract_prompts[start:end]
                        valid = [j for j, img in enumerate(group_generated) if img is not None]
                        if valid:
                            chosen = random.choice(valid)
                            self.image_manager.add(
                                user_prompt=prompts[start],
                                prompt=group_prompts[chosen],
                                sd_pipeline=self.pipe,
                                index=None,
                            )

                    
                    
                    
                    new_reward_list = [
                        float(output_reward_func[idx])
                        if (comp.strip().lower() == "keep")
                        else float(output_reward_func[idx]) - float(original_scores[idx].item())
                        for idx, comp in enumerate(completions)
                    ]

                    rewards_per_func[:, i] = torch.tensor(new_reward_list, dtype=torch.float32, device=device)

                    


        
        
        rewards_per_func = gather(rewards_per_func)

        
        
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        filtered_rewards = [r for r in info.clip_rewards if r != None]
        toxic_rewards = [r for r in info.toxic_rewards if r != None]


        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        if len(filtered_rewards)!=0: self._metrics[mode]["image_reward"].append(torch.tensor(filtered_rewards).mean().item())
        if len(toxic_rewards) != 0:
            self._metrics[mode]["toxic_reward"].append(torch.tensor(toxic_rewards).mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            'pixel_values': pixel_values,
            'image_grid_thws': image_grid_thws,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thws = inputs["image_grid_thws"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, pixel_values, image_grid_thws)

        
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        
        advantages = inputs["advantages"]
        
        
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
            
        per_token_loss[-1] *= torch.exp(per_token_logps[-1]).detach()
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  

        
        
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
