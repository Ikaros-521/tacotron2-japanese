# 一、前言
操作系统：win10
python版本：3.8.13  
安装第三方库：`pip install -r requirements.txt`  
(最恶心的就是 pyopenjtalk，这个需要cmake等才行，反正一堆坑，我也不知道怎么搞好的。。。）
装失败了pycharm提示后补装  
目录How to use里面的2个pt文件需要下载，训练和合成的时候需要用到  
`python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`  

语音合成搭配 How to use中下载的模型文件 然后阅读“启动语音合成.md”，因为我用了conda，所以有一个activate启动虚拟环境。语音合成的配置文件的需求可以看`inference.py`里的说明，默认为 outdir目录下的`config.txt`（存放参数配置）和`data.txt`（存放需要合成的语句）这2个文件，进行批量语音合成。  
`python inference.py`  

## 重要目录说明
`filelists` 存放训练用音频路径和日文文本  
`filelists/ikaros` 存放ikaros音频的训练用文本描述  
`outdir` 输出训练结果路径&语音合成程序配置文件  
`path` 存放数据集文件（音频文件）  


## 训练(无预训练模型)
训练音频要求 
`python train.py --output_directory=outdir --log_directory=logdir`  

## 数据集收集
ikaros部分音频源自 天降之物F 梦幻季节的NDS游戏拆包提取 以及 动漫原声提取降噪。  
使用`NDSTOOL`对nds游戏拆包，提取ahx文件，配合`ahx2mav`程序进行音频转换。
可以参考：[https://blog.csdn.net/Ikaros_521/article/details/126250870](https://blog.csdn.net/Ikaros_521/article/details/126250870)  
由于数据没有文字标注，这边是由群友螺丝开发的基于科大讯飞的一条龙音频文件文本识别程序。  
由于API识别还是有出错的地方，需要手动二次排查结果。  
另外语气词等部分不建议放入训练，所以数据集进行了干扰项的排除。  


另外针对voicestock站点开发了web版本的批量数据爬取。（这个站点收录了很多声优的音频，还有日文标注，不过有的不是很准，也会有杂音，需要优化数据）  
[voicestock 日本声优免费音源素材站点 数据爬取下载](https://www.bilibili.com/video/BV1DP411w7QZ)

## 数据集处理
针对数据集的优化，对音频音量进行均衡。我这开发了以下的python程序。  
[基于ffmpeg开发的多音频文件音量均衡程序](https://ikaros.blog.csdn.net/article/details/128032824)  


# 二、web API搭建

## Ubuntu 无显卡 x86_64 py3.8
安装第三方库：`pip install -r requirements_linux_cpu.txt`  
若cmake安装失败，单独装一下`pip install cmake`  

web API的搭建使用的FastAPI，需要安装下环境 `pip install fastapi uvicorn`  

在本项目路径下，运行`cp -rf waveglow/glow_linux_cpu.py waveglow/glow.py`,覆盖下glow.py源码  

然后后台运行api程序即可(默认监听端口：56789），例如`nohup python3 inference_linux_cpu_api.py > out.log 2>&1 &`  

API调用：`http://<你的服务器公网ip，本机就127.0.0.1>:56789/tt2/content=こにちは`，`こにちは`就是待合成的日文　 
（防火墙端口放行就不用在赘述了吧）  


# 三、Tacotron2-Japanese（官方文档）
- Tacotron2 implementation of Japanese
## Links
* Reference: [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
* [Pre-training tacotron2 models](https://github.com/CjangCjengh/tacotron2-japanese#models)
* [latest changes can be viewed in this repository](https://github.com/StarxSky/tacotron2-JP) 

## How to use（其中2个模型文件需要下载，训练和合成需要用到）
1. Put raw Japanese texts in ./filelists
2. Put WAV files in ./wav
3. (Optional) Download NVIDIA's [pretrained model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing)
4. Open ./train.ipynb to install requirements and start training
5. Download NVIDIA's [WaveGlow model](https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF) or [WaveGlow model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EbyZnGnCJclGl5q_M3KGWTUBq4IIqSLiGznFdqHbv3WM5A?e=8c2aWE) based on Ayachi Nene
6. Open ./inference.ipynb to generate voice

## Cleaners
File ./hparams.py line 30
### 1. 'japanese_cleaners'
#### Before
何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも
#### After
nanikaacltaraitsudemohanashItekudasai.gakuiNnokotojanaku,shijinikaNsurukotodemonanidemo.
### 2. 'japanese_tokenization_cleaners'
#### Before
何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも
#### After
nani ka acl tara itsu demo hanashi te kudasai. gakuiN no koto ja naku, shiji nikaNsuru koto de mo naNdemo.
### 3. 'japanese_accent_cleaners'
#### Before
何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも
#### After
:na)nika a)cltara i)tsudemo ha(na)shIte ku(dasa)i.:ga(kuiNno ko(to)janaku,:shi)jini ka(Nsu)ru ko(to)demo na)nidemo.
### 4. 'japanese_phrase_cleaners'
#### Before
何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも
#### After
nanika acltara itsudemo hanashIte kudasai. gakuiNno kotojanaku, shijini kaNsuru kotodemo nanidemo.

## Models
Remember to change this line in ./inference.ipynb
```python
sequence = np.array(text_to_sequence(text, ['japanese_cleaners']))[None, :]
```
### Sanoba Witch

#### Ayachi Nene 

| Cleaners  Classes | Model |
| ----------- | ----------- |
| japanese_cleaners      | [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/ESltqOvyK3ZPsLMQwpv5FH0BoX8slLVsz3eUKwHHKkg9ww?e=vc5fdd) |
| japanese_tokenization_cleaners   | [Model 2](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/ETNLDYH_ZRpMmNR0VGALhNQB5-LiJOqTaWQz8tXtbvCV-g?e=7nf2Ec) |
|japanese_accent_cleaners| [Model 3](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/Eb0WROtOsYBInTmQQZHf36IBSXmyVd4JiCF7OnQjOZkjGg?e=qbbsv4) |



#### Inaba Meguru

| Cleaners  Classes | Model |
| ----------- | ----------- |
| japanese_tokenization_cleaners | [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/Ed29Owd-E1NKstl_EFGZFVABe-F-a65jSAefeW_uEQuWxw?e=J628nT)|
| japanese_tokenization_cleaners | [Model 2](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/ER8C2tiu4-RPi_MtQ3TCuTkBVRvO1MgJOPAKpAUD4ZLiow?e=ktT81t) |



### Senren Banka
#### Tomotake Yoshino

| Cleaners Classes| Model |
| ----------- | ----------- |
| japanese_tokenization_cleaners| [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EdfFetSH3tpMr7nkiqAKzwEBXjuCRICcvgUortEvE4pdjw?e=UyvkyI)|
| japanese_phrase_cleaners| [Model 2](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EeE4h5teC5xKms1VRnaNiW8BuqslFeR8VW7bCk7SWh2r8w?e=qADqbu)|


#### Murasame

| Cleaners Classes| Model |
| ----------- | ----------- |
| japanese_accent_cleaners| [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EVXUY5tNA4JOqsVL7of8GrEB4WFPrcZPRWX0MP_7G0RXfg?e=5wzBlw)|



### RIDDLE JOKER
#### Arihara Nanami

| Cleaners Classes| Model |
| ----------- | ----------- |
| japanese_accent_cleaners|[Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EdxWxcjx5XdAncOdoTjtyK0BUvrigdcBb2LPmzL48q4smw?e=OlAU66)|

