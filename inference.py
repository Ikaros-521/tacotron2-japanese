import matplotlib
import matplotlib.pylab as plt

import sys

sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser

from scipy.io.wavfile import write
import os

# 配置文件存储于 outdir/config.txt，数据文件存储于 outdir/data.txt
# config.txt文件内容分别为 checkpoint_path（模型路径），waveglow_path（waveglow模型路径），
# data_filename（data文件路径，存放待合成的语句），text_to_sequence_type
my_config = "outdir\config.txt"
# 输出音频目录
# output_dir = ".\\outdir\\wavs"
output_dir = ".\\outdir\\5-105"


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')


hparams = create_hparams()
hparams.sampling_rate = 22050


fo = open(my_config, "a+", encoding='utf-8')
print("读取配置文件: ", fo.name)
# 重新设置文件读取指针到开头
fo.seek(0, 0)
checkpoint_path = ""
waveglow_path = ""
data_filename = ""
text_to_sequence_type = ""
i = 0
for line in fo.readlines():  # 依次读取每行
    text = line.strip()
    if i == 0:
        checkpoint_path = text
    elif i == 1:
        waveglow_path = text
    elif i == 2:
        data_filename = text
    elif i == 3:
        text_to_sequence_type = text
    i = i + 1
# 关闭文件
fo.close()

print("checkpoint_path = " + checkpoint_path)
print("waveglow_path = " + waveglow_path)
print("data_filename = " + data_filename)
print("text_to_sequence_type = " + text_to_sequence_type)

# checkpoint_path = "outdir/checkpoint_24000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()  # .half()

# waveglow_path = 'waveglow_256channels_ljs_v3.pt'
# waveglow_path = '..\waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval()  # .half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

# data_filename = "outdir\data.txt"

fo = open(data_filename, "a+", encoding='utf-8')
print("读取合成音频文件: ", fo.name)
# 重新设置文件读取指针到开头
fo.seek(0, 0)

try:
    os.makedirs(output_dir)
except FileExistsError:
    print(output_dir + "已存在")

for line in fo.readlines():  # 依次读取每行
    text = line.strip()  # 去掉每行头尾空白
    print("本次合成内容为: %s" % (text))

    # text_to_sequence_type = ['japanese_cleaners', 'japanese_accent_cleaners', 'japanese_phrase_cleaners', 'japanese_tokenization_cleaners']
    sequence = np.array(text_to_sequence(text, [text_to_sequence_type]))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    # ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

    audio = denoiser(audio, strength=0.01)[:, 0]

    audio = audio * hparams.sampling_rate
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(output_dir, "{}.wav".format(text))
    write(audio_path, hparams.sampling_rate, audio)
    print('wav输出路径：' + audio_path)

    # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    # ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)

# 关闭文件
fo.close()
