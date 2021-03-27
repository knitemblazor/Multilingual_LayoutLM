import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from layoutlm.modeling.tokenization_bert import BertTokenizer


class TesserOcr:

    def __init__(self, args):
        self.args = args
        self.pdf_path = args.pdf_path
        self.model_path = args.model_name_or_path

    def sliding_window(self,words, tokenizer):
        len_tokens = 0
        split_indices = [0]
        for ind, word in enumerate(words):
            inter = tokenizer.tokenize(word)
            len_tokens += len(inter)
            if len_tokens > 510:
                split_indices.append(ind)
                len_tokens = len(inter)
        split_indices.append(len(words))
        return split_indices

    @staticmethod
    def splitter(start, end, LOI):
        return LOI[start:end]

    def resolver(self, ind, img, pg_width, pg_height, key):
        tokenizer = BertTokenizer.from_pretrained(self.model_path,do_lower_case=True)
        guid_index = 1
        bbox, actual_bbox, text, conf, file_name,page_WH = [], [], [], [], str(ind)+".jpg",[pg_width,pg_height]
        for index,line in enumerate(pytesseract.image_to_data(img,
                                              lang="eng+hin+mal").splitlines()):
            if len(line.split()[6:]) > 5 and index>1:
                xtl, ytl, width, height, conf_, text_ = line.split()[6:]
                xtr = int(xtl) + int(width)
                ybl = int(ytl) + int(height)
                actual_bbox.append([int(xtl), int(ytl), int(xtr), int(ybl)])
                bbox.append([int(1000 * (int(xtl) / page_WH[0])), int(1000*(int(ytl)/page_WH[1])),
                                    int(1000*(int(xtr)/page_WH[0])), int(1000*(int(ybl)/page_WH[1]))])
                text.append(text_)
                conf.append(int(conf_))
        split_indices = self.sliding_window(text,tokenizer)
        inter = []
        for split_index,val in enumerate(split_indices[:-1]):
            bbox_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], bbox)
            actual_bbox_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], actual_bbox)
            text_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], text)
            conf_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], conf)
            inter.append([guid_index,bbox_part,actual_bbox_part,text_part,conf_part,file_name,page_WH])
        return inter

    def loader(self):
        try:
            self.images = convert_from_path(self.pdf_path,dpi=300)
            processed = []
            for img in self.images:
                ratio = img.height / img.width
                size = (1100, int(ratio*1100))
                img.thumbnail(size, Image.ANTIALIAS)
                processed.append(img)
            self.images = processed
        except:
            img = Image.open(self.pdf_path)
            img = img.convert("RGB")
            print("=======",img.size)
            size = (1150, img.height)
            img.thumbnail(size, Image.ANTIALIAS)
            print(img.size)
            self.images = [img]
        page_formatted = []
        for ind, img in enumerate(self.images):
            img = img.convert("RGB")
            for entry in self.resolver(ind, img, width, height, "same"):
                page_formatted.append(entry)
        return page_formatted, key, status


