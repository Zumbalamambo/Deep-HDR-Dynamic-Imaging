from Functions.generate_hdr_utils import *
import Functions.Evaluate as Evaluate

def generate_hdr(model, LDR_images, exposures, label, result_path, scenePath):

    # TODO: Fix for not using in offline mode
    # prepare_input_features(LDR_images, exposures, label)

    inputs = prepare_input_features_offline(LDR_images, exposures, scenePath)
    obj = Evaluate.Evaluate()
    obj.forward_tensorflow(model, inputs)
    # obj.forward_keras(model, inputs)


