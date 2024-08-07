import os
import json
import random
import pandas as pd

random.seed(42) 

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(instructions, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, ensure_ascii=False, indent=4)

def make_instruction(data, json_file, img_dir, kind):
    img_file = data['image'][0]['filename']
    # print(img_file)
    instruction = "<image>\n이 차트를 테이블 정보로 변경해줘."
    gpt_response = create_text_table(data)
    
    instruction = {
        'id': json_file.split(".")[0],
        'image': os.path.join(img_dir, img_file),
        'conversations': [
            {
                "from": "human",
                "value": instruction
            },
            {
                "from": "gpt",
                "value": gpt_response
            }
        ]
    }
    return instruction

def create_text_table(data):
    text_table = ""
    dataframes = []
    image = data["image"][0]
    metadata = data["metadata"]
    chart_sub = metadata["chart_sub"]
    annotations = data["annotations"]
    
    if chart_sub in ["일반 가로 막대형"]:
        annotation = annotations[0]
        categories = annotation['axis_label']['y_axis']
        unit = annotation['unit']
        data_labels = annotation["data_label"]
        legends = annotation["legend"]
        
        if len(legends) == 0:
            df = pd.DataFrame({"분류": categories, f'값 ({unit})': data_labels[0]}) # pdf -> pd 오타 수정
        else:
            df = pd.DataFrame({'분류': categories, f'값 ({unit})' : data_labels[0]})
            for i, legend in enumerate(legends):
                df[f'{legend} ({unit})'] = data_labels[i]
        text_table = annotation["title"] + '\n' + str(df)
        
    elif chart_sub in ["100%기준 누적 가로 막대형", "누적 가로 막대형"]:
        annotation = annotations[0]
        years = annotation['axis_label']['y_axis']
        unit = annotation['unit']
        legends = annotation['legend']
        data_labels = annotation['data_label']
        df = pd.DataFrame({'y축': years})

        # 각 데이터 레이블을 반복하여 새로운 열 추가
        for i, legend in enumerate(legends):
            df[f'{legend} ({unit})'] = data_labels[i]
            
        text_table = annotation["title"] + '\n' + str(df)
            
    elif chart_sub == "방사형":
        annotation = annotations[0]
        categories = annotation['axis_label']['x_axis']
        unit = annotation['unit']
        legends = annotation['legend']
        data_labels = annotation["data_label"]
        if categories != "":
            df = pd.DataFrame({'분류': categories})
        for i, legend in enumerate(legends):
            
            if i < len(data_labels):
                df[f'{legend} ({unit})'] = data_labels[i] + [None] * (len(categories) - len(data_labels[i]))
            else:
                df[f'{legend} ({unit})'] = [None] * len(categories)
        
        text_table = annotation["title"] + "\n" + str(df)
    elif chart_sub == "혼합형":
        for annotation in annotations:
            categories = annotation['axis_label']['x_axis']
            data_label = annotation['data_label'][0]
            unit = annotation['unit']
            
            if annotation["legend"]:
                legend = annotation['legend'][0]
            
                df = pd.DataFrame({
                    '분류': categories,
                    f'{legend} ({unit})': data_label + [None] * (len(categories) - len(data_label))
                })
            else:
                df = pd.DataFrame({
                    '분류': categories,
                    f'값({unit})': data_label + [None] * (len(categories) - len(data)) 
                })
            dataframes.append(df)
        final_df = pd.concat(dataframes, axis=1)
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        text_table = annotations[0]["title"] + "\n" + str(final_df)
    elif chart_sub == "선형":
        annotation = annotations[0]
        categories = annotation['axis_label']['x_axis']
        unit = annotation['unit']
        legends = annotation["legend"]
        data_labels = annotation["data_label"]
        
        df = pd.DataFrame({'분류': categories})
        
        for i, legend in enumerate(legends):
            df[f'{legend} ({unit})'] = data_labels[i]
            
        text_table = data['title'] + '\n' + str(df)
    elif chart_sub in ["100%기준 누적 세로 막대형", "누적 세로 막대형"]:
        annotation = annotations[0]
        categories = data['axis_label']['x_axis']
        unit = data['unit']
        data_labels = data['data_label']
        
        df = pd.DataFrame({'분류': categories})
        
        for i, legend in enumerate(data['legend']):
            df[f'{legend} 비율 ({unit})'] = data_labels[i]
        
        text_table = data["title"] + '\n' + str(df)
    elif chart_sub == "원형":
        annotation = annotations[0]
        labels = annotation['axis_label']['x_axis']
        unit = annotation['unit']
        legends = annotation['legend']
        data_labels = annotation['data_label'][0]
        
        if len(labels) != 0:
            df = pd.DataFrame({'분류': labels, f'비율 ({unit})': data_labels})
        else:
            df = pd.DataFrame({'분류': legends, f'비율 ({unit})': data_labels})
        text_table = annotation["title"] + '\n' + str(df)
    elif chart_sub == "일반 세로 막대형":
        annotation = annotations[0]
        categories = annotation['axis_label']['x_axis']
        unit = annotation['unit']
        data_labels = annotation['data_label']
        if len(data['legend']) != 0:
            df = pd.DataFrame({'분류': categories})

        for i, legend in enumerate(data['legend']):
            df[f'{legend} 값 ({unit})'] = data_labels[i]

        if len(data['legend']) == 0:
            df = pd.DataFrame({'분류': categories, f'값 ({unit})': data_labels[0]})

        text_table = annotation["title"] + '\n' + str(df)
        
    return text_table

def make_data(data_name, target_data_len):
    json_folder_path = "./../../data/train/text_data"
    img_folder_path = "./../../data/train/img_data"

    img_folder_list = sorted(os.listdir(img_folder_path))
    json_folder_list = sorted(os.listdir(json_folder_path))

    instructions = []
    for img_dir, json_dir in zip(img_folder_list, json_folder_list):
        json_dir_path = os.path.join(json_folder_path, json_dir)
        
        
        if json_dir_path.endswith(".py") or json_dir_path.endswith(".txt") or json_dir_path.endswith("mix"):
            continue
        json_files = os.listdir(json_dir_path)
        
        random.shuffle(json_files) 

        # selected_files = json_files[:target_data_len // len(json_folder_list)] 
        selected_files = json_files
        for json_file in selected_files:
            json_file_path = os.path.join(json_dir_path, json_file)
            data = load_data(json_file_path)
            try:
                for kind in ["table"]:
                    instruction = make_instruction(data, json_file, img_dir, kind)
                    if instruction['conversations'][1]['value'] != "" or instruction["conversations"][1]["value"] is not None:
                        instructions.append(instruction)
            except:
                print(f'Error in: {json_file}')

    random.shuffle(instructions)
    # save_data(instructions[:target_data_len * 4], f"./{data_name}.json")
    save_data(instructions, f"./{data_name}.json")


make_data("train_shuffle_all", "all")

