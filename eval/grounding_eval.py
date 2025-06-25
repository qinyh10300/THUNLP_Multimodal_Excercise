import io
import os
import json
import base64
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from mllm.model import MLLMModel
from mllm.model.processing import ModelProcessor
from mllm.model.image_processing import ModelImageProcessor
from utils.file_io import read_jsonlines, read_json



def vis_boxes(img, boxes, expr, save_name='output.png'):
    ### ==> TODO: 可视化Visual Grounding结果，包括给定图像、针对图像中对象的描述和对应对象的坐标框
    pass
    ### <===


    

def eval_model(args):
    model = MLLMModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    img_processor_config = read_json('mllm/model/mllm_preprocessor_config.json')
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)

    model.eval().cuda()

    input_data = read_jsonlines(args.question_file)

    ### TODO: Implement inference loop
    with torch.no_grad():
        correct = total_cnt = 0
        for item in tqdm(input_data):
            image = os.path.join(args.image_dir, item['img_path'])
            expr = item['expression']
            bbox = item['bbox']
            prompt = "Where is {} in image? answer in [x0,y0,x1,y1] format.".format(expr)
            
            msgs = [{"role": "user", "content": prompt}]

            if len(image) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')

            answer = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor
            )

            # Calculate acc
            ### ==> TODO: 实现Visual Grounding的结果准确率计算方法
            pass
            ### <===

            # Visualize VG results
            ### ==> TODO: 实现Visual Grounding结果的可视化
            if args.vis_nums > 0:
                vis_boxes()
                args.vis_nums -= 1
            ### <===

    print(f"Evaluating {args.qannotation_file} ...")
    print(f'Precision @ 1: {correct / total_cnt} \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--sampling", action='store_true')
    parser.add_argument("--vis-nums", type=int, default=5)
    args = parser.parse_args()

    eval_model(args)
