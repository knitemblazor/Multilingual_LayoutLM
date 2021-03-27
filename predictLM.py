from __future__ import absolute_import, division, print_function
import logging
import random
from azure_ocr_2 import AzureOcr
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from layoutlm.modeling.modeling_bert import  BertForTokenClassification
from layoutlm.modeling.configuration_bert import BertConfig
from layoutlm.modeling.tokenization_bert import BertTokenizer
from layoutlm.data.data_loader import read_examples_from_stream
from layoutlm import LayoutlmConfig, LayoutlmForTokenClassification, PayrollDataset
from layoutlm.modeling.file_utils import WEIGHTS_NAME
import re

logger = logging.getLogger(__name__)
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, LayoutlmConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(data):
    batch = [i for i in zip(*data)]
    for i in range(len(batch)):
        if i < len(batch) - 2:
            batch[i] = torch.stack(batch[i], 0)
    return tuple(batch)


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def first(text_tag):
    tags_serial = {0: "S-k_", 1: "B-k_", 2: "I-k_", 3: "E-k_"}
    val = 5
    for tag_id in range(4):
        if re.match(tags_serial[tag_id], text_tag):
            val = tag_id
            break
    return val


def second(comb_list):
    text, tag, bbox, tag_index = "", "", "", 0
    for entry in comb_list:
        text += " " + entry[0]
        for r in (("B-k_", "S-k_"), ("e-k_", "S-k_"), "E-k_", "S-k_"):
            tag = entry[1].replace("B-k_", "S-k_")
        bbox = entry[2]
    return [text.strip(), tag, bbox, tag_index]


def combine(fin_S, fin_oth):
    fin_oth.sort(key=lambda x:int(x[2].split()[0]))
    index_ = []
    for index, item  in enumerate(fin_oth):
        if re.match("B-k_", item[1]):
            index_.append(index)
        if index == len(fin_oth)-1:
            index_.append(index+1)
    for index, val in enumerate(index_[:-1]):
        inter = fin_oth[index_[index]:index_[index+1]]
        inter.sort(key=lambda x:x[3])
        fin_S.append(second(inter))
    fin_S.sort(key=lambda x:int(x[2].split()[0]))

    # print(fin_oth)
    # print(processed)
    print(fin_S)


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    obj_azure = AzureOcr(args)
    file_prefix = args.pdf_name.split("/")[-1]
    processed_list, _, _ = obj_azure.loader()
    tags = ["S-k_", "B-k_", "I-k_", "E-k_"]
    text_label, train_bbox, train_image, text_label_ = [], [], [], []
    file3 = open(args.prediction_file, "w") # text_label
    for page in processed_list:
        examples, words, actual_bboxes, bboxes, file_name, page_size = read_examples_from_stream(page, mode)
        img_id = file_prefix + "_" + file_name
        eval_dataset = PayrollDataset(args, examples, tokenizer, labels, pad_token_label_id, mode=mode)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=1,
            collate_fn=None,
        )
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0].to(args.device),
                    "attention_mask": batch[1].to(args.device),
                    "labels": batch[3].to(args.device),
                }
                if args.model_type in ["layoutlm"]:
                    inputs["bbox"] = batch[4].to(args.device)
                inputs["token_type_ids"] = (
                    batch[2].to(args.device)
                    if args.model_type in ["bert", "layoutlm"]
                    else None
                )
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if args.n_gpu > 1:
                    tmp_eval_loss = (
                        tmp_eval_loss.mean()
                    )
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )
        preds = np.argmax(preds, axis=2)
        label_map = {i: label for i, label in enumerate(labels)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        for i in range(len(words)):
            try:
                if preds_list[0][i] != "O" and preds_list[0][i].split("k_")[-1] != "Table_end":
                    text_label_.append([words[i], bboxes[i], preds_list[0][i]])
                text_label.append(words[i] + "\t" + preds_list[0][i] + "\n")
                train_bbox.append(words[i] + "\t" + str(bboxes[i][0]) + " " + str(bboxes[i][1])
                                  + " " + str(bboxes[i][2]) + " " + str(bboxes[i][3]) + "\n")
                train_image.append(words[i] + "\t" + str(actual_bboxes[i][0]) + " " + str(actual_bboxes[i][1]) + " " +
                                   str(actual_bboxes[i][2]) + " " + str(actual_bboxes[i][3]) + "\t" +str(page_size[0])
                                   + " " + str(page_size[1]) + "\t" + img_id + "\n")
            except:
                pass
    train_ = "".join(i for i in text_label)
    file3.write(train_)
    file3.close()
    fin_S,fin_oth = [], []
    for entry in text_label_:
        if re.match(tags[0],entry[2]):
           fin_S.append([entry[0],entry[2], " ".join(str(i) for i in entry[1]), 0])
        else:
           fin_oth.append([entry[0],entry[2], " ".join(str(i) for i in entry[1]), first(entry[2])])
    combine(fin_S, fin_oth)
    return ""


def predict(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    set_seed(args)
    labels = get_labels(args.labels)
    num_labels = len(labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case
        )
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        df_list = evaluate(
            args, model, tokenizer, labels, pad_token_label_id, mode="test")
    return df_list


