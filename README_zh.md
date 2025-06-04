<!-- ## **TokBench** -->

[English](./README.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/wjf5203/TokBench/refs/heads/homepage/static/images/tokbench_title.jpg"  height=100>
</p>


# TokBench: Evaluating Your Visual Tokenizer before Visual Generation

<div align="center">
  <a href="https://arxiv.org/abs/2505.18142"><img src='https://img.shields.io/badge/arXiv-TokBench-red' alt='Paper PDF'></a>  &ensp;
  <a href="https://wjf5203.github.io/TokBench/"><img src='https://img.shields.io/badge/Project_Page-TokBench-green' alt='Project Page'></a>  &ensp;
  <a href="https://huggingface.co/datasets/Junfeng5/TokBench"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
</div>

<p align="center">
    👋 加入我们的群聊讨论更多Tokenzier！ <a href="assets/wechat.jpg" target="_blank">WeChat</a></a> 
</p>
<p align="center">


-----



👋 欢迎来到TokBench！TokBench是一个专为视觉 Tokenizer和 VAE 设计的评估基准，关注图像重建中的文字和人脸质量，并提供了富含文字和人脸的图片和视频，这种图像细节决定了视觉生成模型的上限。TokBench对现有的图像和视频 Tokenizers/VAEs进行了广泛的评估，**作为挑选VAE/Tokenizer的指南**，参考项目页面 [project page](https://wjf5203.github.io/TokBench/) 查看更多内容。



📈  欢迎大家在issue中提出感兴趣的Tokenizer/VAE，或者直接提供自己尚未开源的 Tokenizer/VAE 的重建结果，我们非常愿意协助评估，并且将结果更新到我们的公开榜单中。如果您已经完成了测试，也欢迎将结果通过邮件发送给我们，我们会在核实真实性后更新到[主页榜单](https://wjf5203.github.io/TokBench/)！





## 🔥🔥🔥 更新!!

* 2025年6月4日: 上传 video set 到HuggingFace。
* 2025年6月4日: 更新video重建评估，和所有论文中方法的 [重建脚本](./tokenzier_vae_scripts/reconstruct.md)。
* 2025年5月27日: 🚀 [Arxiv](https://arxiv.org/abs/2505.18142) 技术报告和[主页](https://wjf5203.github.io/TokBench/)上线。
* 2025年5月16日: 👋 开源 TokBench [Image-Set]()   和 image- level 评估代码。



## **📖 摘要**

视觉Tokenizer和VAE为视觉视觉生成模型提供了高效的压缩视觉隐空间，在带来效率和建模方式提升的同时也引入了压缩损失，影响了视觉生成的上限。通常图像和视频中的文字和人脸具有：1）视觉尺度较小 2）纹理密集丰富 3）重建和生成难度高 4）人类敏感  这几种特性，如果一个Tokenier不能很好的重建文字和人脸，那么视觉生成的上限将会被严重限制。为此TokBench收集了大量富含文字和人脸的图像和视频，并通过OCR文字识别模型和人脸识别模型对文字和人脸重建的效果进行评估，仅需2GB+4分钟即可完成12,000 图片的人脸和文字评估。我们发现传统指标如LPIPS和PSNR对文字和人脸的评估不完全准确，TokBench的指标可以提供更准确的结果帮助大家评估和挑选 Tokenizer 和 VAE。



## 🎉 **与传统指标的差异**

我们发现常用的rFID、LPIPS、PSNR等指标，对文字和人脸的重建效果并不敏感，他们更多关注图像的语义分布和全局信息，对于一些文字和人脸的重建结果会给出跟人类相反的判断：

![metric_compare](assets/metric_compare.jpg)



## 🧱 **TokBench 的Pipeline**

TokBench的评估方式非常简单高效。针对重建的文本，我们提供了这些文本的包围框和准确GT，我们对重建出的图像调用OCR模型直接进行识别，判断重建出的文字是否可被识别。对于人脸，我们调用人脸识别模型提取重建图片和原图的人脸特征，计算特征距离。得益于成熟的工具链[doctr](https://github.com/mindee/doctr)和[insightface](https://github.com/deepinsight/insightface)，我们的pipeline可以在4分钟内完成12000张图片的评估，并且只需要2GB的显存，相比于使用VLM评估的主流方法极大降低了评估负担。

![pipeline](assets/pipeline.jpg)

![metric_visualize](assets/metric_visualize.jpg)





# 📈 实验对比

![main_results](assets/curve.png)

More detailed results and leadborad can be found in [project page](https://wjf5203.github.io/TokBench/).





## 🛠️ 安装和依赖

```shell
# 1. Create conda environment
conda create -n TokBench python==3.10

# 2. Activate the environment
conda activate TokBench

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch -c nvidia

# 4. Install pip dependencies
pip install -e .
pip install nltk
pip install insightface
pip install onnxruntime-gpu
pip install imageio[ffmpeg]


# 5. Install CUDNN for insightface model acceleration
https://developer.nvidia.com/cudnn

# 6. download antelopev2 model for face evaluation following 
https://github.com/deepinsight/insightface/issues/251

```



# 🚀 开始评估

```bash
# 1. download the TokBench data
huggingface-cli download  Junfeng5/TokBench   --repo-type dataset

# 2. reconstruct all images or videos and keep the original folder format like in TokBench 
# Here, refer to the reconstruction of resize baseline
cd tokenzier_vae_scripts/image_scripts
bash resize.sh
cd ../..

# cd tokenzier_vae_scripts/video_scripts
# bash resize.sh
# cd ../..


# 3. Run eval.sh to get the score (T-ACC, T-NED, F-Sim)
bash image_eval.sh
# bash video_eval.sh

```



我们提供了论文中所有tokenizer/VAE的重建脚本，如果需要复现或者参考，可以参考 [重建脚本](./tokenzier_vae_scripts/reconstruct.md)。



## 🔗 BibTeX

如果您认为 [TokBench](https://arxiv.org/abs/2505.18142) 给您的研究和应用带来了一些帮助，可以通过下面的方式来引用:


```BibTeX
 @article{wu2025tokbench,
    title={TokBench: Evaluating Your Visual Tokenizer before Visual Generation}, 
    author={Junfeng Wu and Dongliang Luo and Weizhi Zhao and Zhihao Xie and Yuanhao Wang and Junyi Li and Xudong Xie and Yuliang Liu and Xiang Bai},
    journal={arXiv preprint arXiv:2505.18142},
    year={2025}
  }  
```

