# Forest Fire Detection

The aim of this project is to create neural network models for forest Fire detection using Flask and TensorFlow, and integrate them into a website for convenient use.
> Note: This code is not optimized for production enviroment.

For documentation of the code in this repository, please see the Wiki. See below for instructions to install and run the web server.

# Requirements
- Python => 3.7 and =< 3.10
- pip `sudo apt install python3-pip`
- git `sudo apt install git`

# How to install the code

`git clone git@github.com:akdasUAF/ForestFireDetection.git`

If you do not have access to the repository from your command line, but your github account does have access, then follow the instructions at [this link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to set up ssh keys to provide access. Then, run the `git clone` command again.

### For GSoC 2024 Contributors:
The Deep Belief Network models have been placed in Models/weights. The YOLO model implementation has been stopped for now.
<!--Then, download the three files in [this google drive folder](https://drive.google.com/drive/folders/1aJDFIOFEKJhqdWJ3Ss87bVHyjHAvJaie?usp=sharing) (these are too large for github to accept, and one contains another git repo). Move `yolov5.tar.gz` and `dataset.tar.gz` into the repo directory and extract them using
```
gunzip yolov5.tar.gz
gunzip dataset.tar.gz
tar -xf yolov5.tar
tar -xf dataset.tar
```
Then, move `dbn_pipeline_model.joblib.gz` to the `Models/weights` directory inside of the repo, and extract with

`gunzip dbn_pipeline_model.joblib.gz`
-->
# How to install dependecies

`pip install -r requirements.txt`

If you get warnings, you may need to add the following line to your .bashrc file (the path being added to PATH may vary depending on your operating system):

`export PATH="/home/$USER/.local/bin:$PATH"`

To do this, run

`echo 'export PATH="/home/$USER/.local/bin:$PATH"' >> ~/.bashrc`

Then, close and re-open your terminal or server connection, or just run

`source ~/.bashrc`

# How to run
To run the web server, just do:

`python3 app.py`

If you want it to run persistantly (staying up after you close your terminal connection) do:

`nohup python3 app.py &`
