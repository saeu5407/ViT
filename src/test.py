import os
import torch
import torch.nn as nn

from einops import rearrange
import timm

from models import ViT

if __name__ == '__main__':

    # 공식 레포의 가중치(timm)
    official_model = timm.create_model('vit_base_patch16_384', pretrained=True)
    official_cls = official_model.state_dict()['cls_token']
    official_pos = official_model.state_dict()['pos_embed']
    official_model = nn.Sequential(*list(official_model.children())[:-3])
    official_weight = official_model.state_dict()
    official_weight_key = list(official_weight.keys())

    # 구현 모델
    model = ViT(n_classes=2)
    model = nn.Sequential(*list(model.children())[:-1])
    model_weight_key = list(model.state_dict().keys())

    # pre-train model's weight to ours model
    '''
    [1] official_pos의 차원이 하나 높아서 [0]처리
    [2] timm에는 qkv 레이어로 가중치가 설정되어있으며, 구현한 모델은 query, key, value로 나눠서 설정되어있어서 rearrange 추가 및 if else 처리
    
    [False] 프린트 시 qkv 가 출력되어 있으면 정상
    '''
    model.state_dict()[model_weight_key[0]] = official_cls
    model.state_dict()[model_weight_key[1]] = official_pos[0]
    i = 2
    j = 0
    print("="*50)
    print("="*50)
    while i < len(model_weight_key):
        print(f'[{model.state_dict()[model_weight_key[i]].shape == official_weight[official_weight_key[j]].shape}] '
              f'{model_weight_key[i]} : {official_weight_key[j]}')
        if 'qkv.weight' in official_weight_key[j]:
            qkv_weight = rearrange(official_weight[official_weight_key[j]], '(b x) y -> b x y', b=3)
            model.state_dict()[model_weight_key[i]] = qkv_weight[1] # key : qkv[1]
            model.state_dict()[model_weight_key[i+1]] = qkv_weight[0] # query: qkv[0]
            model.state_dict()[model_weight_key[i+2]] = qkv_weight[2]
            i += 3
        elif 'qkv.bias' in official_weight_key[j]:
            qkv_bias = rearrange(official_weight[official_weight_key[j]], '(b x) -> b x', b=3)
            model.state_dict()[model_weight_key[i]] = qkv_bias[1] # key : qkv[1]
            model.state_dict()[model_weight_key[i+1]] = qkv_bias[0] # query: qkv[0]
            model.state_dict()[model_weight_key[i+2]] = qkv_bias[2]
            i += 3
        else:
            model.state_dict()[model_weight_key[i]] = official_weight[official_weight_key[j]]
            i += 1
        j += 1
    print("="*50)
    print("="*50)

    # 최종 모델 저장
    torch.save(model, f'{os.getcwd().split(os.path.sep + "/")[0]}/pre-trained/vit_pre_train.pth')

    # TODO : 1000개 클래스 기준으로 pre-train 변경