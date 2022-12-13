import os
import torch
import torch.nn as nn

from einops import rearrange
import timm

from models import ViT

if __name__ == '__main__':

    # 공식 레포의 가중치(timm)
    official_model = timm.create_model('vit_base_patch16_384', pretrained=True)
    official_model.eval()
    official_cls = official_model.state_dict()['cls_token']
    official_pos = official_model.state_dict()['pos_embed']
    official_weight = official_model.state_dict()
    official_weight_key = list(official_weight.keys())

    # 구현 모델
    model = ViT(n_classes=1000)
    model.eval()
    model_weight = model.state_dict()
    model_weight_key = list(model_weight.keys())

    # pre-train model's weight to ours model
    '''
    [1] official_pos의 차원이 하나 높아서 [0]처리
    [2] timm에는 qkv 레이어로 가중치가 설정되어있으며, 구현한 모델은 query, key, value로 나눠서 설정되어있으므로 이를 따로 처리
    
    [False] 프린트 시 qkv 가 출력되어 있으면 정상
    '''
    model_weight[model_weight_key[0]] = official_cls
    model_weight[model_weight_key[1]] = official_pos[0]
    i = 2
    j = 2
    print("="*50)
    print("="*50)
    while i < len(model_weight_key):
        print(f'[{model_weight[model_weight_key[i]].shape == official_weight[official_weight_key[j]].shape}] '
              f'{model_weight_key[i]} : {official_weight_key[j]}')
        if 'qkv' in official_weight_key[j]:
            qkv_weight = rearrange(official_weight[official_weight_key[j]], '(b x) y -> b x y', b=3)
            qkv_bias = rearrange(official_weight[official_weight_key[j+1]], '(b x) -> b x', b=3)
            # key: qkv[1]
            model_weight[model_weight_key[i]] = qkv_weight[1]
            model_weight[model_weight_key[i+1]] = qkv_bias[1]
            # query
            model_weight[model_weight_key[i+2]] = qkv_weight[0]
            model_weight[model_weight_key[i+3]] = qkv_bias[0]
            # value
            model_weight[model_weight_key[i+4]] = qkv_weight[2]
            model_weight[model_weight_key[i+5]] = qkv_bias[2]
            i += 6
            j += 2
        else:
            model_weight[model_weight_key[i]] = official_weight[official_weight_key[j]]
            i += 1
            j += 1
    model.load_state_dict(model_weight)
    print("="*50)
    print("="*50)

    print(f"[Warning] Model out check {model(torch.ones([1,3,384,384])) == official_model(torch.ones([1,3,384,384]))}")

    # 최종 모델 저장
    torch.save(model, f'{os.getcwd().split(os.path.sep + "/")[0]}/pre-trained/vit_pre_train.pth')
