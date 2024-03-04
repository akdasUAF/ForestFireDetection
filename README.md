# Our Publication:
- Title: Toward Energy-Efficient Deep Neural Networks for Forest Fire Detection in an Image.
- Link to the Paper: https://digitalcommons.kennesaw.edu/cgi/viewcontent.cgi?article=1051&context=thegeographicalbulletin
> - Citation MLA: Wang, Yali, et al. "Toward Energy-Efficient Deep Neural Networks for Forest Fire Detection in an Image." The Geographical Bulletin 64.2 (2023): 13.
> - Citatin APA: Wang, Y., Purev, C., Barndt, H., Toal, H., Kim, J., Underwood, L., ... & Das, A. K. (2023). Toward Energy-Efficient Deep Neural Networks for Forest Fire Detection in an Image. The Geographical Bulletin, 64(2), 13.
> - Citation BibTex: @article{wang2023toward, title={Toward Energy-Efficient Deep Neural Networks for Forest Fire Detection in an Image}, author={Wang, Yali and Purev, Chuulabat and Barndt, Hunter and Toal, Henry and Kim, Jason and Underwood, Luke and Avalo, Luis and Das, Arghya Kusum}, journal={The Geographical Bulletin}, volume={64}, number={2}, pages={13}, year={2023} }

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

Then, download the three files in [this google drive folder](https://drive.google.com/drive/folders/1cynEIPhHWGcqiry9HhxzSawa3L7VXbTz?usp=drive_link) (these are too large for github to accept, and one contains another git repo). Move `yolov5.tar.gz` and `dataset.tar.gz` into the repo directory and extract them using
```
gunzip yolov5.tar.gz
gunzip dataset.tar.gz
tar -xf yolov5.tar
tar -xf dataset.tar
```
The `dbn_pipeline_model.joblib` has already been extracted to the `Models/weights` directory inside of the repo for use.

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
