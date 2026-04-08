1. NextHAM需要在conda环境下运行：
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda env create -f environment_test.yml
conda activate NextHAM_test
pip install pyatb

2. 阅读test_main.py下的说明，按需调整config.py。

3. python test_main.py，出现Epoch: [0][0/1] loss_ham类似信息则说明成功。可以观察各项loss和mae是否逐渐下降，但不要长时间跑。config.py的预设是train功能，可以尝试自己跑一下test功能。

主程序：train_val.py / test.py
核心网络：nets/nextham.py
