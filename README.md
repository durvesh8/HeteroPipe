Setting up the codebase:

1.	Create a new conda environment with python 3.8
	a.	conda create -n colossal_fin python=3.8
	b.	conda activate colossal_fin
2.	Make sure you have CUDA 11.3 as the main CUDA. Set up a symlink if needed.
	a.	wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
	b.	sudo sh cuda_11.3.0_465.19.01_linux.run
	c.	export PATH="/usr/local/cuda-11.3/bin:$PATH"
	d.	export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
3.	Install Torch 1.12.0 using pip (Not with conda)
	a.	pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
	b.	git clone our repository (Includes the modified colossal AI)
	c.	cd our repo
	d.	pip install -r requirements.txt
	e.	git clone https://github.com/hpcaitech/ColossalAI.git
	f.	pip install packaging
	g.	pip install ninja
	h.	pip install psutil
	i.	(Optional) If you have cached the previously installed extensions, remove them
	i.	rm -rf /home/$USER/.cache/torch_extensions/py38_cu113
	j.	cd ColossalAI-0.2.0
	k.	CUDA_EXT=1 pip install .
	l.	pip install titans==0.0.7
	m.	pip install transformers==4.24.0
