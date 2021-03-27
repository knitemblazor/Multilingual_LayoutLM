import pandas as pd


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x.lower() not in ulist]
    return ulist


def count_occur(test_str, word):
    count = 0
    for i in test_str:
        if i == word:
            count = count + 1
    return count


def duplicate_word_removal(_str):
    l = _str.split()
    k = []
    for i in l:
        if _str.count(i) > 1 and (i not in k) or _str.count(i) == 1:
            k.append(i)
    return ' '.join(k)


def remove(tuples):
    tuples = [t for t in tuples for m in t[0] if t[0][m].strip() != ""]
    return tuples


def desired_labels(label_file):
    label_name = get_labels(label_file)
    desired_label_name = set(
        [i.split('-')[1] for i in label_name if i.split('-')[0] != 'O' if i.split('-')[1].lower().startswith('v')])
    return desired_label_name


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_model_predictions(prediction_file):
    result = []
    with open(prediction_file) as inp:
        for line in inp:
            if len(line.split(' ')) == 3:
                if line.split(' ')[1].lower() != "o" and line.split(' ')[1].split('-')[1][0].lower() != "k":
                    result.append(
                        {"text": line.split(' ')[0], "tag": line.split(' ')[1].replace('\n', '')
                            , "confidence": line.split(' ')[2].replace('\n', '')})

            elif len(line.split('\t')) == 2:
                if line.split('\t')[1] != 'O\n':
                    if line.split('\t')[1].split('-')[1][0].lower() != "k":
                        result.append(
                        {"text": line.split('\t')[0], "tag": line.split('\t')[1].replace('\n', '')
                            , "confidence": str(0.8)})
    return result


def final_pred(model_pred):
    df = pd.DataFrame(model_pred)
    try:
        df['text'] = df[['tag', 'text', 'confidence']].groupby(['tag'])['text'].transform(lambda x: ' '.join(x))
        df_new = df[['tag', 'text']].drop_duplicates().reset_index()
        df_new[['tag_code', 'label_name']] = df_new.tag.str.split('-v_', expand=True)
        aa = df_new.groupby('label_name').agg(lambda x: ' '.join(x))
        final_value = aa['text'].to_dict()
        return final_value
    except:
        return {}


def console_ops_predictions(pred_file):
    res = get_model_predictions(pred_file)
    complete_result = final_pred(res)
    return complete_result


# prediction_file  ="/home/nitheesh/Documents/projects/checkbox_1040/train.txt"
# # path = "/home/nitheesh/Documents/projects/checkbox_1040/layout_lm_model/labels.txt"
# print(console_ops_predictions(prediction_file))
