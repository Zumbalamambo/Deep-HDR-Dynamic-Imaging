# Imports
from Functions.file_io import *
import Functions.init_params as init_params

# Folder name constants
MODEL = 'TrainedModel'
SCENE = 'Scenes'
RESULT = 'Results'

# Read model and initialize all parameters
network = load_model(MODEL)
params = init_params.get_params()

sceneName, scenePath = read_folder(SCENE)
print('Working on scene ------>', sceneName)
result_path = RESULT + '/' + sceneName
make_dir(result_path)

exposures = read_exposure(scenePath)
LDR_images, Label = read_images(scenePath)
