# VDVAE_test

1. Download the dataset FFHQ-256 using https://github.com/openai/vdvae/blob/main/setup_ffhq256.sh on main folder
2. Download the four files in https://github.com/openai/vdvae#ffhq-256 on the folder `pretrained_models/ffhq256`
3. (Optional) Download the model fine-tunned using denoising criterion from https://transfer.sh/MNmIDg/saved_models-1GPU-100h-DC-lowLR.part2.zip on the folder `pretrained_models/ffhq256_DC` (only the files `latest*` are needed).
4. The jupyter notebook `VDVAE_noise.ipynb` contains some code to get started. On cell [4] the model `ffhq256` can be changed to `ffhq256_DC`.
