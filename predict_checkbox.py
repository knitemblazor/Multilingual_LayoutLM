from nnet import Net
import torch
from torch.utils.data import DataLoader
from data_preprocessor import DatasetCheckbox


class ContourCatPredict:
    def __init__(self):
        self.model_path = "/home/nitheesh/Documents/projects_2/checkbox_v2/checkbox_detect/pre_checkbox_model/checkbox_iden.pth"
        self.keys = {0: "false",
                     1: "true"}
        self.batch_size = 30

    def predict(self, feature_vect,label):
        model = Net()
        model.load_state_dict(torch.load(self.model_path))
        test_dataset = DatasetCheckbox(feature_vect,label)
        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        ind = []
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = model(inputs.float())
            for pred_ind in range(self.batch_size):
                try:
                    if list(outputs[pred_ind])[1].item() > .95:
                        ind.append(labels[pred_ind].item())
                except Exception as e:
                    break
        return ind


