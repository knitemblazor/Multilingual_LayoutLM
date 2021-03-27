import requests
import io
from PIL import Image
from pdf2image import convert_from_path
from layoutlm.modeling.tokenization_bert import BertTokenizer


class AzureOcr:

    def __init__(self, args):
        self.args = args
        self.pdf_path = args.pdf_path
        self.model_path = args.model_name_or_path
        self.images = []

    def sliding_window(self, words, tokenizer):
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

    def resolver(self, ind, info_json, pg_width, pg_height, key):
        tokenizer = BertTokenizer.from_pretrained(self.model_path,do_lower_case=True)
        guid_index = 1
        bbox, actual_bbox, text, conf, file_name,page_WH = [], [], [], [], str(ind)+".jpg",[pg_width,pg_height]
        for lines in info_json['lines']:
            for words in lines["words"]:
                xtl, ytl, xtr, ytr, xbr, ybr, xbl, ybl = [int(i) for i in words['boundingBox']]
                if xtr < xtl:
                    a = xtr
                    xtr = xtl
                    xtl = a
                if ybl < ytl:
                    a = ybl
                    ybl = ytl
                    ytl = a
                if key == "same" and words["text"].strip():
                    bbox.append([xtl, ytl, xtr, ybl])
                    actual_bbox.append([int(1000 * (xtl / page_WH[0])), int(1000*(ytl/page_WH[1])),
                                        int(1000*(xtr/page_WH[0])), int(1000*(ybl/page_WH[1]))])
                    text.append(words["text"])
                    conf.append(words["confidence"])
        split_indices = self.sliding_window(text,tokenizer)
        inter = []
        for split_index,val in enumerate(split_indices[:-1]):
            bbox_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], bbox)
            actual_bbox_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], actual_bbox)
            text_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], text)
            conf_part = self.splitter(split_indices[split_index], split_indices[split_index + 1], conf)
            inter.append([guid_index, bbox_part, actual_bbox_part, text_part, conf_part, file_name, page_WH])
        return inter

    @staticmethod
    def get_image_text_data(im_bytes):
        res = requests.post(url='http://104.41.151.78:5010/vision/v3.2-preview.2/read/syncAnalyze',
                                data=im_bytes,
                                headers={'Content-Type': 'application/octet-stream'})
        return res

    def loader(self):
        try:
            # self.images = convert_from_path(self.pdf_path,size=(1100,None), dpi=300, grayscale=True)
            self.images = convert_from_path(self.pdf_path, dpi=300)
            processed = []
            for img in self.images:
                ratio = img.height / img.width
                size = (1100, int(ratio * 1100))
                img.thumbnail(size, Image.ANTIALIAS)
                processed.append(img)
            self.images = processed
        except:
            img = Image.open(self.pdf_path)
            img = img.convert("RGB")
            # size = (1150, img.height)
            # img = img.resize(size, Image.ANTIALIAS)
            self.images = [img]
        page_formatted = []
        for ind, img in enumerate(self.images):
            img = img.convert("RGB")
            b = io.BytesIO()
            img.save(b, 'jpeg')
            im_bytes = b.getvalue()
            res = self.get_image_text_data(im_bytes)
            key = ""
            print("azure response status:", res.status_code)
            if res.status_code == 200:
                status = True
                page = res.json()['analyzeResult']['readResults'][0]
                angle = page['angle']
                width = page['width']
                height = page['height']
                if -20 < angle < 20:
                    key = "same"
                    for entry in self.resolver(ind, page, width, height, "same"):
                        page_formatted.append(entry)
            else:
                pass
        return page_formatted, key, status


