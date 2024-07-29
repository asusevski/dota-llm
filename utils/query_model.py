import cv2
from ascii_magic import AsciiArt
import glob
from pathlib import Path
import cohere

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import warnings

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

def query_nanollava(frame_img: Path):
    # disable some warnings
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings('ignore')

    # set device
    torch.set_default_device('cuda')  # or 'cpu'

    # create model
    model = AutoModelForCausalLM.from_pretrained(
        'qnguyen3/nanoLLaVA',
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        'qnguyen3/nanoLLaVA',
        trust_remote_code=True)

    # text prompt
    prompt = 'What percent of health does the creep have?'

    messages = [
        {"role": "user", "content": f'<image>\n{prompt}'}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)

    # image, sample images can be found in images folder
    image = Image.open(str(frame_img))
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=2048,
        use_cache=True)[0]

    print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())

def query_paligemma(frame_img: Path):
    # disable some warnings
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    warnings.filterwarnings('ignore')

    # set device
    torch.set_default_device('cuda')  # or 'cpu'

    # create model
    model_id = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id)


    # text prompt
    prompt = 'What percent of health does the creep have?'
    image = Image.open(str(frame_img))
    inputs = processor(prompt, image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=20)

    print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])


def query_cohere_for_attack(html_frame: Path, num_lagging_frames: int = 5):
    """
    Query the Cohere API to determine if the hero should attack a creep in the given frame

    Args:
        html_frame (str): path to the frame, converted to html
        num_lagging_frames (int): number of frames to consider before the current frame

    Returns:
        bool: True if the frame contains an attack, False otherwise
    """
    co = cohere.Client(
        api_key=COHERE_API_KEY
    )

    # extract html content for the frame and the lagging frames
    frame_idx = int(html_frame.stem.split('frame')[1])
    html_content = ''
    for _ in range(num_lagging_frames):
        with open(f'frames_html/frame{frame_idx}.html', 'r') as file:
            html_content += file.read()
            frame_idx -= 1
        #with open(f'frames_html/frame{i}.html', 'r') as file:
        #    html_content += file.read()

    message = f"""HTML of last {num_lagging_frames} frames of the game:
    {html_content}
    
    Game state:
    Hero: Drow Ranger, Level 30, Attack Damage 209.

    You are an AI agent in a MOBA game. You are controlling a hero character.
    You are currently in a lane with a creep. The lane status in the last {num_lagging_frames} frames is shown in the above html Should you attack the creep?"""
    chat = co.chat(
        message=message,
        model="command-r-plus"
    )
    return chat