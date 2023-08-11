import pathlib
import sys


# current working directory
path = pathlib.Path().absolute()


## define execution paths
data_path         = "{}\\dataset".format(path)
ml_path         = "{}\\ml_models".format(path)
stack_path         = "{}\\stacking".format(path)
tft_path         = "{}\\temporal_fusion_transformer".format(path)
results_path         = "{}\\results".format(path)

## paste paths to system in order to being able to call modules
sys.path.insert(0, data_path)
sys.path.insert(0, ml_path)
sys.path.insert(0, stack_path)
sys.path.insert(0, tft_path)
sys.path.insert(0, results_path)

##execute dataset construction
exec(open("{}\\construct_dataset.py".format(data_path)).read())

### IMPORTANT: Only uncomment the following three execution statements, when the full pipeline should be run
###            This is not needed as models are already trained on full dataset, read "readme" for more info

##execute base model tuning
#exec(open("{}\\ml_models.py".format(ml_path)).read())

##execute model stacking
#exec(open("{}\\stacking.py".format(stack_path)).read())

##execute temporal fusion transformer
#exec(open("{}\\tft.py".format(tft_path)).read())

## execute results file
exec(open("{}\\results.py".format(results_path)).read())