from fastapi import FastAPI,Request
from fastapi import Response
import json
import msgpack
import io
import os
from PIL import Image

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet,Bottleneck,BasicBlock

app = FastAPI()

@app.get("/ping")
def ping():
    return {"Hello": "World"}

labels_cols = ['redpoint_-3',  'redpoint_-2',  'redpoint_-1',
  'redpoint_0',
  'redpoint_1', 'redpoint_2',   'redpoint_3']

@app.post("/predict")
async def requestPredict(request: Request):
    raw_bin = await request.body()
    data = msgpack.unpackb(raw_bin,raw=False)
    ##メモリからImageobjectに展開
    targetImage =  Image.open(io.BytesIO(data['img'])) 
    w ,h = targetImage.size
    dist = Predict(targetImage,data["remain_ends"],data["last_stone_is_red"],data["red_postion"])
    json_str = json.dumps(dist, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')


target_size = 244

# valid/test用
transform_test = transforms.Compose([
    transforms.Pad(( 240 // 2, 0), fill=0, padding_mode='constant'),  # 左右に余白を追加
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.RandomHorizontalFlip(0.33),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CNNモデルの定義

class myResnet18(models.resnet.ResNet):
    def __init__(self, block,layers,num_classes):
        super().__init__(block,layers,num_classes) # type:ignore
        pass

    def forward(self, xpacked:torch.Tensor) -> torch.Tensor:
        input = xpacked
        #print('input:',input.shape)
        x = input[:,:3]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ###########################################
        params = input[:,3,1,0:3]
        output = torch.cat([x,params],1)
        
        ###########################################
        output = self.fc(output)

        return output

def Predict(cimg:Image.Image,
            remain_ends:int ,
            last_stone_is_red:bool,
            red_postion:int ):
    
    df_test = pd.DataFrame([{'remain_ends':remain_ends,
                          "last_stone_is_red":last_stone_is_red,
                          "red_postion":red_postion }])
    ##print(df_test)
    df_test['remain_ends'] = df_test['remain_ends'].astype(np.float32)
    df_test['last_stone_is_red'] = df_test['last_stone_is_red'].astype(np.float32)
    df_test['red_postion'] = df_test['red_postion'].astype(np.float32)
    # 標準化
    sc = pickle.load(open('./stdsc_02240209.pkl', "rb"))
    df_test[['remain_ends','last_stone_is_red','red_postion']] = sc.transform(df_test)
    
    ## ベストモデル
    fn = 'checkpoint_model.pth'
    net = myResnet18(block=BasicBlock,layers=[2, 2, 2, 2],num_classes=1000)
    ## 最終段交換
        ## 最終を置き換える
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(num_ftrs+3, 1024),
##        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1024, len(labels_cols)),
    )
    net.load_state_dict(torch.load(fn,map_location=torch.device('cpu')))
    
    #zimage = Image.open('sample.png')
    
    device = "cpu"
    gpu_model = net.to(device)
    extend  = df_test[['remain_ends','last_stone_is_red','red_postion']].astype(np.float16).values
    ## 画像は一時保存しないとうまく変換してくれない
    """with tempfile.TemporaryDirectory(dir='./') as dname:
        print(dname)
        fn = os.path.join(dname,'dump.png')
        cimg.save(fn)
        qimage = Image.open(fn)
    
        x = transform_test(qimage)"""
    x = transform_test(cimg)
    ## 次元を足してやってっそこに追加データをぶっこむ
    extend_tensor = np.full((target_size,target_size),255)
    extend_tensor[1][0] = extend[0][0]
    extend_tensor[1][1] = extend[0][1]
    extend_tensor[1][2] = extend[0][2]
    extend_tensor = torch.Tensor(extend_tensor)
    extend_tensor = extend_tensor.unsqueeze(0)
    x = torch.cat([x, extend_tensor], dim=0)
    
    ## [4,244,244] => [1,4,244,244]
    x = x.unsqueeze(0)

    x = x.to(device)
    gpu_model.eval()
    with torch.no_grad():
        y = gpu_model(x)
    y_label = torch.argmax(y, dim=1)
    
    print(y_label)
    y_prob = F.softmax(y,dim=1)
    class_probabilities = y_prob.tolist()
    class_probabilities = class_probabilities[0]
    #print('class_probabilities',class_probabilities)
    def get_top_classes(class_probs, k=3):
        top_classes = []
        for probs in class_probs:
            top_k_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
            top_classes.append(top_k_indices)
        return top_classes    
    y3 = get_top_classes(y)
    #print('top_3:',y3)
    
    response = []
    for n in y3[0]:
        class_name = labels_cols[n]
        prob = class_probabilities[n]
        sender = {'class':n,'acc':prob}
        response.append(sender)
        pass
    
    return response