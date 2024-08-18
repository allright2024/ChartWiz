import os 
import json 
from tqdm import tqdm 
import numpy as np
# train/text_data위치에 맞게 경로 설정
text_dir = "/home/work/ai-hub/data/train/text_data" 
chart_dir_list = os.listdir(text_dir)

class Annotation:
    def __init__(self, annotation, chart_main):
        self.chart_main = chart_main
        self.load_annotation(annotation)
        self.process_data()
    
    def load_annotation(self, annotation):
        self.unit = annotation['unit']
        self.legend = annotation['legend']
        if self.chart_main == "가로 막대형":
            self.x_axis = annotation['axis_label']['y_axis']
            self.y_axis = annotation['axis_label']['x_axis']
        else:
            self.x_axis = annotation['axis_label']['x_axis']
            self.y_axis = annotation['axis_label']['y_axis']
        self.data_label = annotation['data_label']
    
    def process_data(self):
        data_label = np.array(self.data_label)
        len_legend = np.max([1, len(self.legend)])
        len_x_axis = np.max([1, len(self.x_axis)])
        len_y_axis = np.max([1, len(self.y_axis)])
        
        if data_label.shape[1] == len_x_axis:
            self.y_axis=[]
            len_y_axis=1
        elif data_label.shape[1] == len_y_axis:
            self.x_axis=[]
            len_x_axis=1
        else:
            pass
        data_label = data_label.reshape(len_legend, len_x_axis, len_y_axis)
        self.data_label = data_label.squeeze()

class ChartTable:
    def __init__(self, json_data):
        self.load_json(json_data)
    
    def load_json(self, json_data):
        self.filename = json_data["image"][0]["filename"]
        self.width = json_data["image"][0]["width"]
        self.height = json_data["image"][0]["height"]
        self.chart_main = json_data["metadata"]["chart_main"]
        self.chart_sub = json_data["metadata"]["chart_sub"]
        self.title = json_data["annotations"][0]["title"]
        self.description = json_data["description"]
        self.summary = " ".join(json_data["summary"])
        self.annotations = []
        for annotation in json_data["annotations"]:
            self.annotations.append(Annotation(annotation, self.chart_main))

    def make_table(self):
        table = f"TITLE | {self.title}\n"
        for n, annotation in enumerate(self.annotations):
            if len(annotation.legend) == 0:
                for x, data in zip(annotation.x_axis, annotation.data_label):
                    table += f"{x} | {data}{annotation.unit}\n"
            else:
                if n==0:
                    for x in annotation.x_axis:
                        table += f"| {x} "
                    table += "\n"
                for i, legend in enumerate(annotation.legend):
                    table += f"{legend} "
                    if len(annotation.legend)==1:
                        for data in annotation.data_label:
                            table += f"| {data}{annotation.unit} "
                    else:
                        for data in annotation.data_label[i]:
                            table += f"| {data}{annotation.unit} "
                    table += "\n"
                
           
        return table              
    
json_path_list = []
table_json_list = []

for chart_dir in chart_dir_list:
    chart_dir_path = os.path.join(text_dir, chart_dir)
    json_list = os.listdir(chart_dir_path)
    for json_file in tqdm(json_list):
        json_path = os.path.join(chart_dir_path, json_file)
        with open(json_path, "r") as f:
            json_data = json.load(f)
        try:
            chart_table = ChartTable(json_data)
            table = chart_table.make_table()
            table_json = {}
            table_json["id"] = json_file 
            table_json["image"] = json_data["image"][0]["filename"]
            conv_1_json = {
                "from": "human",
                "value": "<image>\n이 차트를 테이블 정보로 변경해줘."
            }
            conv_2_json = {
                "from": "gpt",
                "value": table
            }
            table_json["conversations"] = [conv_1_json, conv_2_json]
            table_json_list.append(table_json)
        
        except:
            json_path_list.append(json_path)
            