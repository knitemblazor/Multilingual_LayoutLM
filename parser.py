from predictLM import predict
import yaml
from objectify import Struct


class CheckboxParser:
    def __init__(self, pdf_path, pdf_name):
        self.config_path = "config.yaml"
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name

    def checkbox_parser(self):
        with open(self.config_path) as file:
            model_config = yaml.load(file)
            model_info_dict = {}
            for key in model_config.keys():
                model_info_dict.update(model_config[key])
        args = Struct(**model_info_dict)
        args.pdf_path = self.pdf_path
        args.pdf_name = self.pdf_name
        df_processed = predict(args)
        # df_processed.to_csv("df_trial.csv", index=False)
        return df_processed


