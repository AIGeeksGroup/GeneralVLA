



## Environment 
```
conda create -n seg python==3.9 -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirement.txt
pip install gradio==3.39.0
pip install numpy==1.26.4
```

### model
```

mkdir pretrain_model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

git clone https://huggingface.co/xinlai/LISA-13B-llama2-v1-explanatory

cd segagent
git clone git clone https://www.modelscope.cn/zzzmmz/SegAgent-Model.git

cd simpleclick
download cocolvis_vit_large.pth from: https://drive.google.com/drive/folders/1zVhZefCjsTBxvyxnYMVnbkrNeRCH6y9Y
```

## Deployment

```

sh run_demo.sh


```
```
Please input your prompt: Please segment the cap part of the bottle.

Please input the image path: img/openjar.png

Please input your prompt: Please segment the handle part of the umbralla.

Please input the image path: img/takeumbrella.png

Please input your prompt: Please segment the Rubik's cube.

Please input the image path: img/sort.png

```

By default, we use 4-bit quantization. Feel free to delete the `--load_in_4bit` argument for 16-bit inference or replace it with `--load_in_8bit` argument for 8-bit inference.


