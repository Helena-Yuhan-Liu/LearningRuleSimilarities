from multisys.neural_datasets.Mante2013.mante2013 import MANTE13_DATASET_STANDARD, MANTE13_DT, Mante13Dataset
# from multisys.neural_datasets.Siegel2015.siegel15 import Siegel15Dataset, SIEGEL15_DATASET_SUBJECT1
# from multisys.neural_datasets.Siegel2015.siegel2015 import Siegel15Dataset

def init_neural_dataset(task_name, dt):
    if task_name in ['Mante13', 'VisualMante', 'OriginalMante']:
        if dt == MANTE13_DT: # standard dt for Mante task is 50, but this might be set as something else in the config.
            return MANTE13_DATASET_STANDARD # this is already pre-initialized to save time
        else:
            return Mante13Dataset(dt=dt) # otherwise, need to re-initialize with different dt
    if task_name in ['Siegel15', 'VisualSiegel']:
        if dt == 50:
            return SIEGEL15_DATASET_SUBJECT1
        else:
            return Siegel15Dataset(dt=dt)
    else:
        raise NotImplementedError # TODO: add other datasets