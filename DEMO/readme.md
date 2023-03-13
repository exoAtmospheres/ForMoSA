## DEMO tutorial

(Works when the github repository is cloned.)
Here we assume you created a conda environment following the general instructions.

***
***

# OPTION 1

Copy the config (*.ini) file from the example and edit it as needed. 

Adapt your observations file to the right format following the inputs/create_obs_fitsfile.py example.

You can follow interactive_formosa.ipynb to perform a modeling and plot the outcomes. 


***
***

# OPTION 2

You can lunch formosa from your terminal as following. 

You still need to clone the config (*.ini) file, adapt it to your case and save your data in the right format. 

***
$ conda activate formosa_env

$ python /path_to_folder/ForMoSA/ForMoSA/main.py ‘/path_to_ini_file/demo.ini’ y 


(the "y" at the end of the command line stands for doing the interpolation of the atmospheric grid, if the grid is already interpolated, then change "y" -> "n")

***

To plot and work with the outcomes please check the "output_plots.ipynb" example at the outputs folder of this DEMO.
