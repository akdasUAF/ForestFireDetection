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

Then, download the two files in [this google drive folder](https://drive.google.com/drive/folders/1aJDFIOFEKJhqdWJ3Ss87bVHyjHAvJaie?usp=sharing) (these are too large for github to accept, and one contains another git repo) and move them into the repo directory. Then extract them using
```
gunzip yolov5.tar.gz
gunzip dataset.tar.gz
tar -xf yolov5.tar
tar -xf dataset.tar
```

# How to install dependecies

`pip install -r requirements.txt`

If you get warnings, you may need to add the following line to your .bashrc file (the path being added to PATH may vary depending on your operating system):

`export PATH="/home/$USER/.local/bin:$PATH"`

To do this, run

`echo 'export PATH="/home/$USER/.local/bin:$PATH"' >> ~/.bashrc`

Then, close and re-open your terminal or server connection, or just run

`source ~/.bashrc`

# Create DBN model files
`python3 db_create_train.py`

# How to run
`python3 -m flask run`
 
