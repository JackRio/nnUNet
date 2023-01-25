#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.paths import *


class ExperimentPlanner3D_v21_Teacher(ExperimentPlanner3D_v21):
    """
    """

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21_Teacher, self).__init__(folder_with_cropped_data,
                                                              preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_Teacher"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_Teacher_plans_3D.pkl")


#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.paths import *


class ExperimentPlanner3D_v21_Student(ExperimentPlanner3D_v21):
    """
    """

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21_Student, self).__init__(folder_with_cropped_data,
                                                              preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_Student"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_Student_plans_3D.pkl")
        self.unet_base_num_features = 16
