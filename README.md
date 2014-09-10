SpotTheHotSpot
==============

#### Using this software
This software is released under the GPLv2 license.
If you use this software in a scientific publication, please cite the paper:

`TBD bibtex reference`

## Matlab version

#### Requirements
* Image Processing Toolbox
* Signal Processing Toolbox

The Signal Processing Toolbox is needed just for the `findpeaks()` function. Writing a custom, toolbox-independent version of it should not be that hard. 

## C++ version

#### Requirements
* [OpenCV](http://opencv.org/)

#### Instructions
1. `cd src`
2. `make`
3. `./main [videofile]`

## Pyhton version (Stitching)

#### Requirements
* [OpenCV](http://opencv.org/)

#### Instructions
1. `./main_stitching_asift.py [videofile]`
