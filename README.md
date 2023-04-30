# ForestFireDetection

The aim of this project is to create Fire detection NN model with Flask and TensorFlow
> Note: This code is not optimized for production enviroment.

# Requirement
- Python => 3.7 and =< 3.10
- pip `sudo apt install python3-pip`
- git `sudo apt install git`

# How to install dependecies

`pip install -r requirements.txt`

If you get warnings, you may need to add the following line to your .bashrc file (the path being added to PATH may vary depending on your operating system):

`export PATH="/home/$USER/.local/bin:$PATH"`

# How to install the code

`git clone git@github.com:akdasUAF/ForestFireDetection.git`

Then, download the two files in [this google drive folder](https://drive.google.com/drive/folders/1aJDFIOFEKJhqdWJ3Ss87bVHyjHAvJaie?usp=sharing) (these are too large for github to accept, and one contains another git repo) and move them into the repo directory. Then extract them using
```
gunzip yolov5.tar.gz
gunzip dataset.tar.gz
tar -xf yolov5.tar
tar -xf dataset.tar
```

# How to run
`python -m flask run`
 
