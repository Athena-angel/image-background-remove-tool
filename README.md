# Tool to remove the background from the portrait using Tensorflow
A tool to remove a background from a portrait image using Tensorflow
**********************************************************************
### Setup:
* Clone repository ```git clone https://github.com/OPHoperHPO/image-background-remove-tool```
* Run ```./bin/setup.sh``` _This setup.sh script loads the trained model._
**********************************************************************
### Dependencies:
```	tensorflow, pillow, tqdm, numpy, scipy ```
**********************************************************************
### Running the script:
 * Put images to the input folder.
 * Run ```run.sh``` for Linux or ```run.bat``` for Windows
 
 > Note:  _You can remove ```1``` in the ``` run.sh (bat) ```script to speed up image processing, but the quality will be worse_
**********************************************************************
### Differences from the [original script](https://github.com/susheelsk/image-background-removal):
* Added comments to the code.
* Added ```tqdm``` progress bar.
* __Removes background from image without loss of image resolution.__
* __The script now processes all images from the input folder and saves them to the output folder with the same name.__
* __New sample images.__
**********************************************************************
### TODO:
```
1) Add a graphical interface. (0% done)
```
### Sample Result:
* __More sample images in input and output folders__
* Input: 
* ![alt text](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/input/1.jpg "Input")

* Output: 
* ![alt text](https://github.com/OPHoperHPO/image-background-remove-tool/blob/master/output/1.png "Output")
**********************************************************************
