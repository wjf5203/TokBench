<!-- ## **TokBench** -->

[中文阅读](./README_zh.md)

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
    👋 Join our <a href="assets/wechat.jpg" target="_blank">WeChat</a></a> 
</p>
<p align="center">


-----

 

👋 **Welcome to TokBench** TokBench is an benchmark specifically designed for visual Tokenizers and VAEs, focusing on text and face quality in image reconstruction. It provides a rich collection of text- and face-centric images and videos, as these fine-grained details determine the upper bound of visual generation models. TokBench conducts extensive evaluations of existing image and video Tokenizers/VAEs, **serving as a guide for selecting VAE/Tokenizers**. Refer to the  [project page](https://wjf5203.github.io/TokBench/)  for more details.





📈  ***Feel free to suggest more Tokenizers/VAEs*** you're interested in via issues, or directly share reconstruction results of your unreleased Tokenizer/VAE. We're happy to assist with evaluation and update our public [leaderboard](https://wjf5203.github.io/TokBench/) with your results. If you've already completed testing, you're welcome to email us your results - we'll verify and update them on our leaderboard! 



## 🔥🔥🔥 News!!

- Jun 4, 2025: Update video set on HuggingFace.
- Jun 4, 2025: Updated video evaluation scripts and [reconstruction scripts](./tokenzier_vae_scripts/reconstruct.md) for all methods in the paper.
- May 27, 2025: 🚀 [Arxiv](https://arxiv.org/abs/2505.18142)  technical report and [project page](https://wjf5203.github.io/TokBench/) launched.
- May 16, 2025: 👋 Open-sourced TokBench [Image-Set]()  and image-level evaluation code.



## **📖 **Abstract****

Visual tokenizers and VAEs have significantly advanced visual generation and multimodal modeling by providing more efficient compressed or quantized image representations. However, while helping production models reduce computational burdens, the information loss from image compression fundamentally limits the upper bound of visual generation quality. To evaluate this upper bound, we focus on assessing reconstructed text and facial features since they typically: 1) exist at smaller scales, 2) contain dense and rich textures, 3) are prone to collapse, and 4) are highly sensitive to human vision. We first collect and curate a diverse set of clear text and face images from existing datasets. Unlike approaches using VLM models, we employ established OCR and face recognition models for evaluation, ensuring accuracy while maintaining an exceptionally lightweight assessment process **requiring just 2GB memory and 4 minutes** to complete. Using our benchmark, we analyze text and face reconstruction quality across various scales for different image tokenizers and VAEs. Our results show modern visual tokenizers still struggle to preserve fine-grained features, especially at smaller scales. We further extend this evaluation framework to video, conducting comprehensive analysis of video tokenizers. Additionally, we demonstrate that traditional metrics fail to accurately reflect reconstruction performance for faces and text, while our proposed metrics serve as an effective complement.



## 🎉 **Comparison with Previous Metrics**

Our research found that commonly used metrics like rFID, LPIPS, and PSNR are not sensitive enough to evaluate text and face reconstruction quality. These metrics primarily focus on semantic distribution and global image information, sometimes producing judgments that contradict human perception when assessing text and face reconstruction results.

![metric_compare](assets/metric_compare.jpg)





## 🧱 **TokBench Pipeline**

TokBench's evaluation is remarkably simple yet effective. For reconstructed text, we provide bounding boxes and precise ground truth (GT), then directly apply OCR models to the reconstructed images to determine whether the text remains recognizable. For faces, we employ face recognition models to extract facial features from both reconstructed and original images, then calculate feature distances. Leveraging mature toolchains like [doctr](https://github.com/mindee/doctr) and [insightface](https://github.com/deepinsight/insightface), our pipeline can evaluate 12,000 images in just 4 minutes while requiring only 2GB of GPU memory - significantly reducing evaluation overhead compared to mainstream VLM-based assessment methods. 

![pipeline](assets/pipeline.jpg)

![metric_visualize](assets/metric_visualize.jpg)





# 📈 Comparisons

![main_results](assets/curve.png)

More detailed results and leadborad can be found in [project page](https://wjf5203.github.io/TokBench/).



## 🛠️ Dependencies and Installation

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

# 6. auto-download antelopev2 model and move the ckpts for face evaluation following 
[https://github.com/deepinsight/insightface/issues/251](https://github.com/deepinsight/insightface/issues/2766)

```



# 🚀 Evaluating your Tokenizer/VAE

```bash
# 1. download the TokBench data
huggingface-cli download  Junfeng5/TokBench   --repo-type dataset

# 2. reconstruct all images and keep the original folder format like in TokBench 
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



We provide reconstruction scripts for all tokenizers/VAEs in the paper. If you need to reproduce or refer to them, you can refer to [reconstruction scripts](./tokenzier_vae_scripts/reconstruct.md).



## 🔗 BibTeX

If you find [TokBench](https://arxiv.org/abs/2505.18142) useful for your research and applications, please cite using this BibTeX:


```BibTeX
 @article{wu2025tokbench,
    title={TokBench: Evaluating Your Visual Tokenizer before Visual Generation}, 
    author={Junfeng Wu and Dongliang Luo and Weizhi Zhao and Zhihao Xie and Yuanhao Wang and Junyi Li and Xudong Xie and Yuliang Liu and Xiang Bai},
    journal={arXiv preprint arXiv:2505.18142},
    year={2025}
  }  
```

