Make root dir and `cd` into it
```
git clone https://github.com/Lux-AI-Challenge/Lux-Design-2021.git
git clone git@github.com:core2duo/lux_ai.git

./lux_ai/build.sh
./lux_ai/run.sh
```

Run visualization:
```
docker run -it lux_ai serve -d dist
```

Run a single game inside the container with:
```
lux-ai-2021 lux_ai/baseline.py lux_ai/baseline.py --python=python3 --maxtime=5000
```

Run infinite games:
```
python3 lux_ai/solution/agents.py -n 5
```
where 5 is simultaneous matches count


# Running docker inside docker
Go to sdc root directory and checkout `f/SDC-54454-cuda-11.1`.
Run `./d -suya` and `build`.

Make root lux_ai dir inside ~/sdc directory:
```
mkdir ~/sdc/lux_ai
```
Clone repos as usual inside it.

Then run following commands:
```
mkdir -p ~/sdc/tmp/lux_ai
ln -s ~/sdc/tmp/lux_ai /tmp/lux_ai
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update < /dev/null
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo apt-get install libcudnn8
pip3 install -U pip
pip3 install -r ~/sdc/lux_ai/lux_ai/requirements.txt
```
Tensorboard:
```
python3 ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir /tmp/tb_logs/ --port 6007
```