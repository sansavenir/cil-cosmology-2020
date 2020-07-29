Model used for generating galaxy images.

For the model to work, first the training data needs to be generated.
```
python prep_data.py --dataDir path_to_training_data
```
This will create the stars/ and coords/ directories and fill them with the stars and their coordinates respectively.
It will also save the background values inside of the backgroung.npy file.  
Afterwards, the wasserstein Gan that will generate the stars has to be trained. 
Can be done inside of image_gen/.
```
python gan.py
```
The same can then be done for the coordinate generator inside of coord_gen/.  
Once both of the generators are trained, we can procede in generating the galaxy images.
```
python generate_galaxies.py --imageModel path_to_star_generator --coordModel path_to_coordinate_generator
```
The final images will be saved in results/.