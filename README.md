# ExpGest

🔥(ICME 2024) ExpGest: Expressive Speaker Generation Using Diffusion Model and Hybrid Audio-Text Guidance

[Arxiv](https://arxiv.org/abs/2410.09396) [Paper](https://www.computer.org/csdl/proceedings-article/icme/2024/10687922/20F0zy3t2py)

> *IEEE International Conference on Multimedia and Expo (ICME), 2024*

This is the official repository of the ExpGest.

https://github.com/ExpGest/assets/6f722731-5acf-45c2-aafd-b2f6f76b2058


ExpGest is a method that accepts audio, phrases, and motion description text as inputs, and based on a diffusion model, it generates highly expressive motion speakers.



# News :triangular_flag_on_post:

- [2025/03/13] **Code release!** ⭐
- [2024/10/12] **ExpGest is on [arXiv](https://arxiv.org/abs/2410.09396) now.**
- [2024/03/08] **ExpGest got accepted by ICME 2024!** 🎉

## To-Do list 📝
- [x] Inference code
- [ ] Training code
      
## Requirements 🎉

### Conda environments
```
conda create -n ExpGest python=3.7
conda activate ExpGest 
conda install -n ExpGest pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Pre-trained model and data
- Download [CLIP](https://drive.google.com/drive/folders/1CN9J2T1tN-F2R5qfHjOfMkGXP00oka6E?usp=drive_link) model. Put the folder in the root.
- Download pre-trained weights from [here](https://drive.google.com/drive/folders/1GNGsOKTJf6GSrp9OENi0AmA2UkrwLx7u?usp=drive_link) (only audio-control) and put it in `./mydiffusion_zeggs`
- Download pre-trained weights from [here](https://drive.google.com/drive/folders/175TyMLMjzXz5vkCHOmvB9v7YxXcs40Te?usp=drive_link) (action-audio-control) and put it in `./mydiffusion_zeggs`
- Download pre-trained weights from [here](https://drive.google.com/drive/folders/1_l3LMxYZvyWGjn9D9qQVdbPkmClDfI5K?usp=drive_link) (text-audio-control) and put it in `./mydiffusion_zeggs`
- Download WavLM weights from [here](https://drive.google.com/drive/folders/1du41ziM0utAMjCtn-YPM8ZYOI6YplHrq?usp=drive_link) and put it in `./mydiffusion_zeggs`
- Download ASR weights from [here](https://drive.google.com/drive/folders/1lUi6QrpxH03xHSsO-__bW_I3eXPzrZ1A?usp=drive_link) and put it in `./mydiffusion_zeggs`
- Download BERT weights from [here](https://drive.google.com/drive/folders/1lLU3SLMHBjIeiELYtH6uBqkGqhRYpFPC?usp=drive_link) and put it in `./mydiffusion_zeggs`


## Run 

Users can modify the configuration file according to their own needs to achieve the desired results. Instructions for modification are provided as comments in the configuration file.

```
# More detailed controls such as texts and phrases, please set them in the configuration file.
# Run audio-control demo:
python sample_demo.py  --audiowavlm_path '../1_wayne_0_79_79.wav' --max_len 320 --config ../ExpGest_config_audio_only.yml

# Run hybrid control demo:
python sample_demo.py  --audiowavlm_path '../1_wayne_0_79_79.wav' --max_len 320 --config ../ExpGest_config_hybrid.yml

# Run demo with emotion guided:
python sample_demo.py  --audiowavlm_path '../1_wayne_0_79_79.wav' --max_len 320 --config ../ExpGest_config_hybrid_w_emo.yml
```





## Citation
```
@inproceedings{cheng2024expgest,
  title={ExpGest: Expressive Speaker Generation Using Diffusion Model and Hybrid Audio-Text Guidance},
  author={Cheng, Yongkang and Liang, Mingjiang and Huang, Shaoli and Ning, Jifeng and Liu, Wei},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgement
The pytorch implementation of ExpGest is based on [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture). We use some parts of the great code from [FreeTalker](https://github.com/YoungSeng/FreeTalker) and [MLD](https://github.com/ChenFengYe/motion-latent-diffusion). We thank all the authors for their impressive works!!

## Contact
For technical questions, please contact cyk990422@gmail.com

