# AB3DMOT Installation

First of all, clone the code:
~~~shell
git clone https://github.com/xinshuoweng/AB3DMOT.git
~~~

## System Requirements

This code has only been tested on the following combination of major pre-requisites. Please check beforehand.

* Ubuntu 18.04
* Python 3.6

## Dependencies:

This code requires the following packages:
1. scikit-learn==0.19.2
2. filterpy==1.4.5
3. numba==0.43.1
4. matplotlib==2.2.3
5. pillow==6.2.2
6. opencv-python==4.2.0.32
7. glob2==0.6
8. PyYAML==5.4
9. easydict==1.9
10. llvmlite==0.32.1 			
11. wheel==0.37.1

One can either use the system python or create a virtual enviroment (venv for python3) specifically for this project (https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv). To install required dependencies on the system python, please run the following command at the root of this code:
```
cd path/to/AB3DMOT
pip3 install -r requirements.txt
```
To install required dependencies on the virtual environment of the python, please run the following command at the root of this code:

```
pip3 install venv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

Additionally, this code depends on my personal toolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox. Please install the toolbox by:

*1. Clone the github repository.*
~~~shell
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
~~~

*2. Install dependency for the toolbox.*
~~~shell
cd Xinshuo_PyToolbox
pip3 install -r requirements.txt
cd ..
~~~

Please add the path to the code to your PYTHONPATH in order to load the library appropriately. For example, if the code is located at /home/user/workspace/code/AB3DMOT, please add the following to your ~/.profile:
```
export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:/home/user/workspace/code/AB3DMOT/Xinshuo_PyToolbox
```
Then update your configuration with
```
source ~/.profile
cd path/to/AB3DMOT
source env/bin/activate
```
You are now done with the installation! Feel free to play on the supported datasets ([KITTI](docs/KITTI.md), [nuScenes](docs/nuScenes.md))!