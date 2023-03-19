from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.paths import *

class ExperimentPlanner3D_v21_Base(ExperimentPlanner3D_v21):
    """
        Experiment planner for Student network in the Teacher-Student architecture
    """

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21_Base, self).__init__(folder_with_cropped_data,
                                                              preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_Base"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_Base_plans_3D.pkl")
        self.unet_base_num_features = 16