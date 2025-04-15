import random
import argparse
import json
import transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments
from peft import PeftModel
from finetune import MAX_LENGTH
from finetune import replace_at_index, interleave_lists_with_zeros, find_all_curve_end_positions, count_curve
import pickle
import os
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def infill_type(input_str, mask_type):
    positions_sk = find_all_curve_end_positions(input_str, "<sketch_end>")
    positions_ex = find_all_curve_end_positions(input_str, "<extrusion_end>")
    interleave_lists = interleave_lists_with_zeros(positions_sk, positions_ex)
    lists_len = len(interleave_lists[:-1])
    if mask_type != 'extrusion':
        i = random.choice(range(lists_len))
        while i % 2 != 0:
            i = random.choice(range(lists_len))
        if i == 0:
            start_index_sk = 0
        else:
            start_index_sk = interleave_lists[i] + 16
        gt_sk = input_str[start_index_sk: interleave_lists[i + 1]]
        if mask_type == 'es':
            prompt = (
                'Below is a partial description of a CAD sequence where one '
                'command has been replaced with the string "[sketch-extrusion mask]":\n'
            )
            mask_str = replace_at_index(input_str, start_index_sk, interleave_lists[i + 2] + 15,
                                        '[sketch-extrusion mask]')
            infill_str = prompt + mask_str + "\n"
            infill_str += (
                "Generate a string that could replace \"[sketch-extrusion mask]\" in the CAD sequence:\n"
            )
        elif mask_type == 'sketch':
            prompt = (
                'Below is a partial description of a CAD sequence where one '
                'command has been replaced with the string "[sketch mask]":\n'
            )
            mask_str = replace_at_index(input_str, start_index_sk,
                                        interleave_lists[i + 1] + 12, '[sketch mask]')
            infill_str = prompt + mask_str + "\n"
            infill_str += (
                "Generate a string that could replace \"[sketch mask]\" in the CAD sequence:\n"
            )
        elif mask_type == 'face':
            prompt = (
                'Below is a partial description of a CAD sequence where one '
                'command has been replaced with the string '
            )
            multi_mask = '[face mask] ' * gt_sk.count('face_end')
            prompt += '\"' + multi_mask[:-1] + "\".\n"
            mask_str = replace_at_index(input_str, start_index_sk,
                                        interleave_lists[i + 1], multi_mask)
            infill_str = prompt + mask_str + "\n"
            infill_str += (
                "Generate a string that could replace "
            )
            infill_str += '\"' + multi_mask[:-1] + "\"" + ' in the CAD sequence:\n'
        else:
            local_sketch = input_str[start_index_sk:interleave_lists[i + 1]]
            face_end_index = find_all_curve_end_positions(local_sketch, "<face_end>")
            face_end_index.insert(0, 0)
            j = random.choice(range(len(face_end_index[:-1])))
            if j == 0:
                start_index_j = 0
            else:
                start_index_j = face_end_index[j] + 11

            gt_face = local_sketch[start_index_j:face_end_index[j + 1] - 1]  # local face, without face_end

            if mask_type == 'loop':
                num_local_loop = gt_face.count('loop_end')
                multi_loop_mask = '[loop mask] ' * num_local_loop
                local_sketch_mask = replace_at_index(local_sketch, start_index_j,
                                                     face_end_index[j + 1], multi_loop_mask)
                mask_str = replace_at_index(input_str, start_index_sk,
                                            interleave_lists[i + 1], local_sketch_mask[:-1] + ' ')
                prompt = (
                    'Below is a partial description of a CAD sequence where one '
                    'command has been replaced with the string '
                )
                prompt += '\"' + multi_loop_mask[:-1] + "\".\n"

                infill_str = prompt + mask_str + "\n"
                infill_str += (
                    "Generate a string that could replace "
                )
                infill_str += '\"' + multi_loop_mask[:-1] + "\"" + ' in the CAD sequence:\n'

            else:
                local_face = gt_face
                loop_end_index = find_all_curve_end_positions(local_face, "<loop_end>")
                loop_end_index.insert(0, 0)
                k = random.choice(range(len(loop_end_index[:-1])))
                if k == 0:
                    start_index_k = 0
                else:
                    start_index_k = loop_end_index[k] + 11

                gt_loop = local_face[start_index_k:loop_end_index[k + 1] - 1]  # local loop, without loop_end
                if mask_type == 'lac':
                    multi_curve_mask = count_curve(gt_loop)
                elif mask_type == 'curve':
                    num_local_curve = gt_loop.count('curve_end')
                    multi_curve_mask = '[curve mask] ' * num_local_curve
                local_face_mask = replace_at_index(local_face, start_index_k,
                                                   loop_end_index[k + 1], multi_curve_mask)
                local_sketch_mask = replace_at_index(local_sketch, start_index_j,
                                                     face_end_index[j + 1], local_face_mask + ' ')
                mask_str = replace_at_index(input_str, start_index_sk,
                                            interleave_lists[i + 1], local_sketch_mask[:-1] + ' ')
                prompt = (
                    'Below is a partial description of a CAD sequence where one '
                    'command has been replaced with the string '
                )
                prompt += '\"' + multi_curve_mask[:-1] + "\".\n"

                infill_str = prompt + mask_str + "\n"
                infill_str += (
                    "Generate a string that could replace "
                )
                infill_str += '\"' + multi_curve_mask[:-1] + "\"" + ' in the CAD sequence:\n'

    else:
        i = random.choice(range(lists_len))
        while i % 2 != 1:
            i = random.choice(range(lists_len))
        prompt = (
            'Below is a partial description of a CAD sequence where one '
            'command has been replaced with the string "[extrusion mask]":\n'
        )
        mask_str = replace_at_index(input_str, interleave_lists[i] + 12,
                                    interleave_lists[i + 1] + 15, ' [extrusion mask]')

        infill_str = prompt + mask_str + "\n"
        infill_str += (
            "Generate a string that could replace \"[extrusion mask]\" in the CAD sequence:\n"
        )
    return infill_str



def prepare_model_and_tokenizer(args):
    if args.model_name=="8B":
        model_id = "meta-llama/Meta-Llama-3-8B"
        print(f"Model size: {model_id}")
        pipeline = transformers.pipeline("text2text-generation",
                                         model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map='auto')
        tokenizer = pipeline.tokenizer
        model = pipeline.model
    else:
        llama_options = args.model_name.split("-")
        is_chat = len(llama_options) == 2
        model_size = llama_options[0]

        def llama2_model_string(model_size, chat):
            chat = "chat-" if chat else ""
            return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

        model_string = llama2_model_string(model_size, is_chat)
        print(f"Model size: {model_string}")
        model = LlamaForCausalLM.from_pretrained(
            model_string,
            load_in_8bit=True,
            device_map="auto",
        )

        tokenizer = LlamaTokenizer.from_pretrained(
            model_string,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")

    return model, tokenizer


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    prompts = []
    originals = []
    for _ in range(args.num_samples):
        if args.mask_type == 'unconditional':
            prompt = 'Below is a description of a CAD sequence:\n'
            input_str = None
        elif args.mask_type == 'cad':
            prompt = 'Below is a partial description of a CAD sequence where one command has been replaced with the string "[sketch-extrusion mask]".\n[sketch-extrusion mask] \nGenerate a string that could replace "[sketch-extrusion mask]" in the CAD sequence:\n'
            input_str = None
        elif args.mask_type == 'curve' and args.use_fixed_demo:
            prompt = 'Below is a partial description of a CAD sequence where one command has been replaced with the string "[line mask] [line mask] [linemask] [line mask]":\nline,18,2 <curve_end> line,44,2 <curve_end> line,44,60 <curve_end> line,18,60 <curve_end> <loop_end> [line mask] [line mask] [line mask] [line mask] <loop_end> <face_end> <sketch_end> add,31,34,31,31,31,1,0,0,0,0,1,0,-1,0,48,31,31 <extrusion_end>\nGenerate a string that could replace "[line mask] [line mask] [line mask] [line mask]" in the CAD sequence:\n'
            input_str = None
        else:
            with open(args.data_path, "rb") as f:
                data_test = pickle.load(f)
            i = random.randint(0, len(data_test)-1)
            input_str = data_test[i]
            prompt = infill_type(input_str, args.mask_type)
        if not (args.mask_type == 'extrusion' or args.mask_type == 'curve'):
            list_c = ['circle','line','arc']
            j = random.randint(0, 2)
            prompt = prompt + list_c[j]
        prompts.append(prompt)
        originals.append(input_str)

    outputs = []
    delimiter = "in the CAD sequence:"
    while len(outputs) < len(prompts):
        batch_prompts = prompts[len(outputs): len(outputs) + args.batch_size]

        batch = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}
        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=MAX_LENGTH,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if args.mask_type == 'es':
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1]
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                prompt = prompt.replace('[sketch-extrusion mask]', gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'sketch':
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1]
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                prompt = prompt.replace('[sketch mask]', gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'extrusion':
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1]
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                prompt = prompt.replace('[extrusion mask]', gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'face':
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1]
                face_number = gen_str.count('face_end')
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                multi_face = '[face mask] '*face_number
                prompt = prompt.replace(multi_face, gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'loop':
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1] + ' '
                loop_number = gen_str.count('loop_end')
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                multi_loop = '[loop mask] '*loop_number
                prompt = prompt.replace(multi_loop, gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'curve' and not args.use_fixed_demo:
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1] + ' '
                curve_number = gen_str.count('curve_end')
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                multi_curve = '[curve mask] '*curve_number
                prompt = prompt.replace(multi_curve, gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'curve' and args.use_fixed_demo:
            for i in range(len(gen_strs)):
                gen_str = gen_strs[i].split(delimiter, 2)[1] + ' '
                prompt = batch_prompts[i]
                prompt = prompt.split('\n')[1]
                multi_curve = '[line mask] [line mask] [line mask] [line mask] '
                prompt = prompt.replace(multi_curve, gen_str).replace('\n', '')
                outputs.append(prompt)
        elif args.mask_type == 'unconditional':
            try:
                for i in range(len(gen_strs)):
                    parts = gen_strs[0].split("a CAD sequence:\n")[1]

                    outputs.append(parts)
            except:
                continue
        elif args.mask_type == 'cad':
            try:
                for i in range(len(gen_strs)):
                    parts = gen_strs[0].split("in the CAD sequence:\n")[1]
                    outputs.append(parts)
            except:
                continue
        else:
            outputs.extend(gen_strs)
        print(f"Generated {len(outputs)}/{len(prompts)}samples.")
        with open(os.path.dirname(args.model_path)+'conditional_samples_'+str(args.num_samples)+'_'+args.mask_type+'_mask.json', "w") as f:
            for prompt, output, original in zip(prompts, outputs, originals):
                f.write(json.dumps({"prompt": prompt, "output": output, "original":original}) + "\n")
    with open(os.path.dirname(args.model_path)+'conditional_samples_'+str(args.num_samples)+'_'+args.mask_type+'_mask.json', "w") as f:
        for prompt, output, original in zip(prompts, outputs, originals):
            f.write(json.dumps({"prompt": prompt, "output": output, "original":original}) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="8B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--out_path", type=str, default="_cad_samples.json")
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mask_type", type=str, default="sketch")
    parser.add_argument("--use_fixed_demo", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    conditional_sample(args)
