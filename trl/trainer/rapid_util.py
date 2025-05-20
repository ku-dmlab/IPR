import hashlib
import base64
import os
import random
from PIL import Image
from qwen_vl_utils import process_vision_info
def generate_image(sd_pipeline, prompt: str):
        result = sd_pipeline(prompt, num_inference_steps=1, guidance_scale=0.0)
        return result.images[0]
    
def make_input(prompt, image_manager, pipe, processor):
    image_file_path, index = image_manager.sample(prompt, pipe)
    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "image",
                    "image": image_file_path,
                },
                {"type": "text", "text": f"""
    <prompt> You are an AI model designed to refine prompts for a text-to-image generation model. Your goal is to ensure that the refined prompt produces an image that aligns more accurately with the **user's original intent**, rather than the provided image.

    Your response must be formatted as follows:
    <prompt>
    refined prompt
    </prompt>

    If the given image is already well-aligned with the user's original prompt, respond with:
    <prompt>
    Keep
    </prompt>

    Otherwise, refine the prompt to optimize the output so that the text-to-image model generates an image that better matches the user's original prompt. Do not modify the prompt to fit the provided image if it deviates from the original intent.

    The goal is to reduce misalignment between the generated image and the original prompt, ensuring that the final output is as close as possible to what the user intended.

    prompt: {prompt}
    """},
            ],
        }
    ]
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs
class ImageManager:
    def __init__(self, base_path='./images'):
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
        
        # Generate new image
        image = generate_image(sd_pipeline, prompt)
        
        # Determine the file name
        if index is None:
            # Find the next available index
            existing_files = self._get_image_files(user_prompt)
            indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('image_')]
            index = max(indices) + 1 if indices else 0
        
        # Save the image
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
            # No images found, generate a new one if sd_pipeline is provided
            if sd_pipeline is None:
                raise ValueError("No images found for this prompt and no pipeline provided to generate one")
            return self.add(prompt, sd_pipeline)
        
        if index is not None:
            # Try to get the specific index
            target_file = f"image_{index}.png"
            if target_file in image_files:
                image_file_path = os.path.join(self._get_prompt_dir(prompt), target_file)
                return image_file_path, index
            else:
                raise ValueError(f"No image with index {index} found for this prompt")
        
        # Select a random image
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



