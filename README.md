# Expressive_Gesture
🔥(ICME 2024) ExpGest: Expressive Speaker Generation Using Diffusion Model and Hybrid Audio-Text Guidance

This is the official repository of the ExpGest.

![image](https://github.com/cyk990422/ExpGesture/blob/main/9.png)


# News :triangular_flag_on_post:

- [2025/03/13] **Code release!** ⭐
- [2024/10/12] **ExpGest is on [arXiv]([https://arxiv.org/abs/2303.05938](https://arxiv.org/abs/2410.09396)) now.**
- [2024/03/08] **ExpGest got accepted by ICME 2024!** 🎉
## Requirements

### Conda environments
```
conda create -n ExpGest python==3.8.8  
conda activate ExpGest 
conda install -n ExpGest pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Pre-trained model and data
- Download [CLIP](https://drive.google.com/drive/folders/1CN9J2T1tN-F2R5qfHjOfMkGXP00oka6E?usp=drive_link) model. Put the folder in the root.
- Download pre-trained weights from [here](https://drive.google.com/file/d/1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9/view?usp=share_link) (only audio-control) and put it in `./main/mydiffusion_zeggs`
- Download pre-trained weights from [here]([https://drive.google.com/file/d/1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9/view?usp=share_link](https://drive.google.com/drive/folders/175TyMLMjzXz5vkCHOmvB9v7YxXcs40Te?usp=drive_link) (action-audio-control) and put it in `./main/mydiffusion_zeggs`
- Download pre-trained weights from [here]([https://drive.google.com/file/d/1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9/view?usp=share_link](https://drive.google.com/drive/folders/1_l3LMxYZvyWGjn9D9qQVdbPkmClDfI5K?usp=drive_link) (text-audio-control) and put it in `./main/mydiffusion_zeggs`
- Download WavLM weights from [here]([[https://drive.google.com/file/d/1aCeKMVgIPqYjafMyUJsYzc0h6qeuveG9/view?usp=share_link](https://drive.google.com/drive/folders/1_l3LMxYZvyWGjn9D9qQVdbPkmClDfI5K?usp=drive_link) and put it in `./main/mydiffusion_zeggs` 

## Demo

```
# Run a real-time demo:
python -m acr.main --demo_mode webcam -t

# Run on a single image:
python -m acr.main --demo_mode image --inputs <PATH_TO_IMAGE>

# Run on a folder of images:
python -m acr.main --demo_mode folder -t --inputs <PATH_TO_FOLDER> 

# Run on a video:
python -m acr.main --demo_mode video -t --inputs <PATH_TO_VIDEO> 
```















# For Beats datasets
Note: All FGD evaluation models are trained in a way similar to the trimodal model, using linear velocity representation as input. Among them, "body" refers to the evaluation using only all the torso joints from the upper body above the spine 3 to the left and right wrists in SMPLX.
"Full" represents the evaluation with an additional 30 finger joints.
"EA" stands for emotion - alignment. A sentiment classifier is trained using UNet. Its input is the generated motion sequence, and the output is a label corresponding to one of the 8 emotions in the Beat data.
The accuracy of this classifier on the speaker 1 sequence of the Beat data is 95.7%.
It is used to determine whether the motion sequences generated by our ExpGes conform to the emotional categories of the original data.


|method|FGD_full on raw|FGD_full on feature|FGD_body on raw|FGD_body on feature|EA body|EA hand|
|---|---|---|---|---|---|---|
|Ground Truth|0.0|0.0|0.0|0.0|0.97|0.96|
|CAMN|-|-|52.4|263.9|-|-|
|Trimodal|-|-|47.6|212.7|-|-|
|DiffuseStyleGesture|52.7|303.9|33.7|133.9|0.60|0.49|
|ExpGes w/o emo-guided|40.6|199.1|14.8|84.9|0.68|0.55|
|ExpGes|25.3|115.0|11.7|76.6|0.91|0.88|
|ExpGes + hybrid|44.6|226.1|25.4|129.3|0.89|0.82|

"EC" represents the success rate of emotion - control. Similarly, the trained sentiment classifier is used to perform gradient - guided emotion control. Then, the final motion sequences generated by the emotion - guidance module + ExpGes are subjected to emotion judgment to determine whether they conform to the artificially given emotions.

|method|EA body|EC body|EA hand|EC hand|
|---|---|---|---|---|
|DiffuseStyleGesture|0.60|0.27|0.49|0.19|
|DiffuseStyleGesture w emo-guided|0.83|0.69|0.70|0.63|
|ExpGes|0.91|0.83|0.81|0.70|

#user study
|method|Human-likeness|Appropriateness|Expressive|Consistency|
|---|---|---|---|---|
|DiffuseStyleGesture|3.86|3.81|3.34|-|
|ExpGes w/o emo-guided|3.92|3.99|3.78|-|
|ExpGes|4.17|4.22|4.01|-|
|ExpGes + long motion|3.15|3.32|3.59|2.96|
|ExpGes + hybrid|4.27|4.11|3.73|4.07|

#ablation study
|method|FGD_full on raw|FGD_full on feature|Human-likeness|Appropriateness|
|---|---|---|---|---|
|DiffuseStyleGesture (baseline)|52.7|303.9|3.86|3.81|
|ExpGes (ours) w/o sem-align|42.8|225.3|3.78|3.74|
|ExpGes (ours) w/o emo-guided|40.6|199.1|3.92|3.99|
|Full model|25.3|115.0|4.17|4.22|


🎉**Code coming soon**

