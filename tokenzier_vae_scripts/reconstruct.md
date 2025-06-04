We provide the reconstruction code of tokenizer/vae for all images/videos in the paper here, including the code of resize baseline. You can directly use resize baseline to verify your evaluation environment, or download the code and weights of other tokenizer/vae to reproduce our evaluation. 



For the image_scripts and video_scripts directories, if you want to use methods other than resize, please git clone the corresponding repositories of these methods so that the directories are as follows, and manually change the weight addresses corresponding to these models.

Expected Directory Structure：

```bash
-image_scripts
	-chameleon
	-SEED-Voken
	-taming-transformers
	-TokenFlow
	-UniTok
	-VAR
	-vavae_tokenizer
	-resize.py
	-var_rec.py
	-...
-video_scripts
	-Step-Video-T2V
	-HunyuanVideo
	-Cosmos-Tokenizer
	-CogVideo
	-resize.py
	-hunyuan_rec.py
	-...
```





You can directly use the **resize.sh** script or **llamagen.sh**， **cosmos.sh** script as a reference example, and build your own tokenizer reconstruction script based on it.