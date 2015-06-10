1) How to compile
Change directory to project 
Use visual studio to open the project.vcxproj 
Change the "Additional Include Directories" of C/C++ and CUDA C/C++ of the project properties if the directory of the installed CUDA header file is not the same as the one in the project
Build the project and the binary program "project.exe" will be copied to directory simulation

2) How to run command line (with inputs if necessary) 
In the command window, change directory to simulation 
Run the command: .\project.exe [-d distance] [-a angle] [-f dataset]
[-d distance]: parse distance_threshold through program argument, by default the value is 1
[-a angle]: parse angle_threshold through program argument, by default the value is 0.99999
[-f dataset]: parse dataset file through program argument, by default the value is "dataset"

3) Expected results
The number of points of original dataset file simulation\dataset will be reduced 
The result points of GPU algorithm will be written to file simulation\dataset_gpu.txt
The result points of CPU algorithm will be written to file simulation\dataset_c.txt

