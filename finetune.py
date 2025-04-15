import os
import argparse
import torch
import random
from pathlib import Path
import pickle
from dataclasses import dataclass
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

accelerator = Accelerator()
IGNORE_INDEX = -100
MAX_LENGTH = 512
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class CADDataset(Dataset):
    def __init__(self, pickle_fn, llama_tokenizer=None):
        if not os.path.exists(pickle_fn):
            raise ValueError(f"{pickle_fn} does not exist")
        self.inputs = pickle.load(open(pickle_fn, "rb"))
        self.llama_tokenizer = llama_tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self.inputs):
            raise ValueError(f"Index {index} out of range")
        val = self.inputs[index]
        val = self.tokenize(val)
        return val

    def tokenize(self, input_str):
        if random.random() < 0.12:
            tokens, answer_token, val_len = self.generation_task(input_str)
        else:
            tokens, answer_token, val_len = self.infill_task(input_str)
        input_ids = labels = tokens.input_ids[0]
        input_id_lens = label_lens = (
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()
        )
        return dict(
            input_ids=input_ids,
            input_id_lens=input_id_lens,
            labels=labels,
            label_lens=label_lens,
            val_len=val_len,
        )

    def generation_task(self, input_str):

        prompt = "Below is a description of a CAD sequence:\n"
        prompt += input_str
        answer = input_str
        answer_token = self.llama_tokenizer(
            answer + self.llama_tokenizer.eos_token,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        val_len = self.llama_tokenizer(
            (prompt + self.llama_tokenizer.eos_token).split('CAD sequence:\n')[-1],
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        ).input_ids.shape[1] - 1
        tokens = self.llama_tokenizer(
            prompt + self.llama_tokenizer.eos_token,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        return tokens, answer_token, val_len

    def infill_task(self, input_str):
        all_type = ['cad','es','extrusion','sketch','face','loop','curve','cla'] # cla: circle, line, arc
        mask_type = random.choice(all_type)
        if mask_type == 'es' and (input_str.count('sketch_end') == 1):
            mask_type = 'sketch'
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
            if mask_type == 'cad':
                prompt = (
                    'Below is a partial description of a CAD sequence where one '
                    'command has been replaced with the string '
                )
                multi_mask = '[sketch-extrusion mask] ' * input_str.count('sketch_end')
                prompt += '\"' + multi_mask[:-1] + "\".\n"
                mask_str = replace_at_index(input_str, 0,
                                            len(input_str), multi_mask)
                infill_str = prompt + mask_str + "\n"
                infill_str += (
                    "Generate a string that could replace "
                )
                infill_str += '\"' + multi_mask[:-1] + "\"" + ' in the CAD sequence:\n'
                infill_str += input_str
                answer = input_str
            elif mask_type == 'es' and (input_str.count('sketch_end')>1):
                prompt = (
                    'Below is a partial description of a CAD sequence where one '
                    'command has been replaced with the string "[sketch-extrusion mask]":\n'
                )
                mask_str = replace_at_index(input_str, start_index_sk, interleave_lists[i + 2] + 15,
                                            '[sketch-extrusion mask]')
                gt_es = input_str[start_index_sk:interleave_lists[i + 2] + 15]
                infill_str = prompt + mask_str + "\n"
                infill_str += (
                    "Generate a string that could replace \"[sketch-extrusion mask]\" in the CAD sequence:\n"
                )
                infill_str += gt_es
                answer = gt_es
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
                infill_str += input_str[start_index_sk: interleave_lists[i + 1] + 12]
                answer = input_str[start_index_sk: interleave_lists[i + 1] + 12]
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
                infill_str += gt_sk
                answer = gt_sk
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
                    infill_str += gt_face
                    answer = gt_face

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
                    if mask_type == 'cla':
                        multi_curve_mask = count_curve(gt_loop)
                    else:
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
                    infill_str += gt_loop
                    answer = gt_loop

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

            infill_str = prompt + mask_str  + "\n"
            infill_str += (
                "Generate a string that could replace \"[extrusion mask]\" in the CAD sequence:\n"
            )
            infill_str += input_str[interleave_lists[i] + 13: interleave_lists[i + 1] + 15]
            answer = input_str[interleave_lists[i] + 13: interleave_lists[i + 1] + 15]
        answer_token = self.llama_tokenizer(
            answer + self.llama_tokenizer.eos_token,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        val_len  = self.llama_tokenizer(
            (infill_str + self.llama_tokenizer.eos_token).split('in the CAD sequence:\n')[-1],
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        ).input_ids.shape[1] - 1
        tokens = self.llama_tokenizer(
            infill_str + self.llama_tokenizer.eos_token,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )

        return tokens, answer_token, val_len

def count_curve(loop):
    c_line = find_all_curve_end_positions(loop, "line")
    c_arc = find_all_curve_end_positions(loop, "arc")
    c_circle = find_all_curve_end_positions(loop, 'circle')
    dict_curve_type = {}
    for c_i in c_line:
        dict_curve_type[c_i] = '[line mask] '
    for c_i in c_arc:
        dict_curve_type[c_i] = '[arc mask] '
    for c_i in c_circle:
        dict_curve_type[c_i] = '[circle mask] '
    list_curve = ''
    c_all = c_line + c_arc + c_circle
    c_all.sort()
    for c_i in c_all:
        list_curve += (dict_curve_type[c_i])
    return list_curve

def find_all_curve_end_positions(string,curve_end):
    positions = []
    start = 0
    while True:
        position = string.find(curve_end, start)
        if position == -1:
            break
        positions.append(position)
        start = position + len(curve_end)
    return positions

def interleave_lists_with_zeros(list1, list2):
    result = [0]
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        result.append(list1[i])
        result.append(list2[i])
    result.extend(list1[min_length:])
    result.extend(list2[min_length:])
    return result

def replace_at_index(original_string, index, index_h, replacement):
    if index < 0 or index >= len(original_string):
        return "Index out of range"
    return original_string[:index] + replacement + original_string[index_h:]

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances]
            for key in ("input_ids", "labels")
        )
        val_len = [instance['val_len'] for instance in instances][0]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        if labels.shape[1] < MAX_LENGTH:
            labels[:,:labels.shape[1] - val_len] = IGNORE_INDEX
            # decoded_text = self.tokenizer.decode([token1 for token1 in labels[0] if token1 != -100])
        else:
            labels[:,:] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_datasets(args, llama_tokenizer):
    datasets = {
        "train": CADDataset(
            str(args.data_path / "train.pkl"),
            llama_tokenizer=llama_tokenizer,
        ),
        "val": CADDataset(
            str(args.data_path / "val.pkl"),
            llama_tokenizer=llama_tokenizer,
        ),
    }

    return datasets


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not args.fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        label_names=["cad_ids"],  
    )
    return training_args


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


def setup_model(args, rank):
    if args.model_name=="8B":
        model_id = "meta-llama/Meta-Llama-3-8B"
        print(f"Model size: {model_id}")
        pipeline = transformers.pipeline("text2text-generation",
                                         model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map={"": rank})
        llama_tokenizer = pipeline.tokenizer
        model = pipeline.model
    elif args.model_name=="70B":
        model_id = "meta-llama/Meta-Llama-3-70B"
        print(f"Model size: {model_id}")
        pipeline = transformers.pipeline("text2text-generation",
                                         model="meta-llama/Meta-Llama-3-70B", load_in_8bit=True, model_kwargs={"torch_dtype": torch.bfloat16},
                                         device_map={"": rank})
        llama_tokenizer = pipeline.tokenizer
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
            load_in_8bit=args.fp8,
            device_map={"": rank},
        )

        llama_tokenizer = LlamaTokenizer.from_pretrained(
            model_string,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer

def setup_trainer(args):
    training_args = setup_training_args(args)
    model, llama_tokenizer = setup_model(args, training_args.local_rank)
    datasets = setup_datasets(args, llama_tokenizer)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer


def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default="exp")
    parser.add_argument("--model-name", default="8B")
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--data-path", type=Path, default="data/basic")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=500, type=int)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "CADLLM"
    print(args)
    main(args)
