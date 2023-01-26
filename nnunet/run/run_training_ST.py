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


import argparse
from copy import deepcopy

import wandb
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.paths import default_plans_identifier
from nnunet.run.default_configuration import get_default_configuration
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def wandb_config(trainer):
    wandb.config.training_nodules = len(trainer.dataset_tr)
    wandb.config.validation_nodules = len(trainer.dataset_val)
    wandb.config.batch_size = trainer.batch_size
    wandb.config.patch_size = trainer.patch_size
    wandb.config.initialized = trainer.was_initialized
    wandb.config.initial_lr = trainer.initial_lr
    wandb.config.patience = trainer.patience


def wandb_init(args):
    """
        Initialize wandb config and runs
    """
    # Wandb Init
    try:
        with open("/mnt/netcache/bodyct/experiments/subsolid_nodule_segm_nnunet/wandb_key.txt") as f:
            key = f.read()
        wandb.login(key=key)
    except FileNotFoundError:
        print("Key not found")

    wandb.init(id=args.wandb_name, project="subsolid_segmentation_nnunet", entity="sanyog_v", name=args.wandb_name,
               tags=[args.task, args.network_trainer, args.p], resume=True)

    trainer_kwargs = json.loads(args.trainer_kwargs.replace("\\", ""))
    for key, val in trainer_kwargs.items():
        wandb.config.update({f"{key}": f"{val}"}, allow_val_change=True)
    wandb.config.network = args.network
    wandb.config.network_trainer = args.network_trainer
    wandb.config.plans = args.p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument('--student_kwargs', required=False, default="{}",
                        help="Use a dictionary in string format to specify keyword arguments. This will get"
                             " parsed into a dictionary, the values get correctly parsed to the data format"
                             " and passed to the trainer. Example (backslash included): \n"
                             r"--student_kwargs {\"teacher_impact\": 0.8, \"network\": \"3d_fullres\",\"network_trainer\": \"nnUNetTrainerV2_Student\",\"plans_identifier\":\"nnUNetData_plans_v2.1_Student\"}")
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument('--trainer_kwargs', required=False, default="{}",
                        help="Use a dictionary in string format to specify keyword arguments. This will get"
                             " parsed into a dictionary, the values get correctly parsed to the data format"
                             " and passed to the trainer. Example (backslash included): \n"
                             r"--trainer_kwargs {\"class_weights\":[0,2.00990337,1.42540704,2.13387239,0.85529504,0.592059,0.30040984,8.26874351],\"weight_dc\":0.3,\"weight_ce\":0.7}")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument('--wandb_name', type=str, required=True, help='Unique wandb name for the run')

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    # Checking for student network
    student_kwargs = json.loads(args.student_kwargs.replace("\\", ""))
    if not len(student_kwargs):
        # TODO: More cleaner response later
        raise Exception("Student kwargs not provided")

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        teacher_class = get_default_configuration(network, task, network_trainer, plans_identifier)
    plans_file_s, output_folder_name_s, dataset_directory_s, batch_dice_s, stage_s, \
        student_class = get_default_configuration(student_kwargs['network'], task, student_kwargs['network_trainer'],
                                                  student_kwargs['plans_identifier'])

    if teacher_class is None:
        raise RuntimeError("Could not find teacher class in nnunet.training.network_training")

    if student_class is None:
        raise RuntimeError("Could not find student class in nnunet.training.network_training")

    assert issubclass(teacher_class, nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"
    assert issubclass(student_class, nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer_teacher = teacher_class(plans_file, fold, output_folder=output_folder_name,
                                    dataset_directory=dataset_directory,
                                    batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                    deterministic=deterministic,
                                    fp16=run_mixed_precision, **json.loads(args.trainer_kwargs.replace("\\", "")))

    trainer_student = student_class(plans_file_s, fold, output_folder=output_folder_name_s,
                                    dataset_directory=dataset_directory_s, batch_dice=batch_dice_s, stage=stage_s,
                                    unpack_data=decompress_data, deterministic=deterministic,
                                    fp16=run_mixed_precision, teacher_impact=student_kwargs["teacher_impact"],
                                    **json.loads(args.trainer_kwargs.replace("\\", "")))

    if args.disable_saving:
        trainer_teacher.save_final_checkpoint = False  # whether or not to save the final checkpoint
        trainer_teacher.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer_teacher.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer_teacher.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

        # Student Network
        trainer_student.save_final_checkpoint = False  # whether or not to save the final checkpoint
        trainer_student.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer_student.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer_student.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer_teacher.initialize(not validation_only)
    trainer_student.initialize(not validation_only)
    trainer_student.dl_tr = deepcopy(trainer_teacher.dl_tr)
    trainer_student.dl_val = deepcopy(trainer_teacher.dl_val)
    trainer_student.tr_gen = deepcopy(trainer_teacher.tr_gen)
    trainer_student.val_gen = deepcopy(trainer_teacher.val_gen)

    # Initialize wandb and update config
    wandb_init(args=args)
    wandb_config(trainer_teacher)
    wandb.config.postprocessing_on_folds = not disable_postprocessing_on_folds

    if find_lr:
        trainer_teacher.find_lr()
        trainer_student.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer_teacher.load_latest_checkpoint()
                trainer_student.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                # TODO: Pretrained weights param in student kwargs
                load_pretrained_weights(trainer_teacher.network, args.pretrained_weights)
                if student_kwargs.get("pretrained_weights", None):
                    load_pretrained_weights(trainer_student.network, student_kwargs['pretrained_weights'])
                else:
                    print("No pre-trained weights specified for student network")
            else:
                # new training without pretrained weights, do nothing
                pass

            # TODO: Do this part
            trainer_teacher.run_training()

        else:
            if valbest:
                trainer_teacher.load_best_checkpoint(train=False)
                trainer_student.load_best_checkpoint(train=False)
            else:
                trainer_teacher.load_final_checkpoint(train=False)
                trainer_student.load_final_checkpoint(train=False)

        trainer_teacher.network.eval()
        trainer_student.network.eval()

        # predict validation
        trainer_teacher.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                                 run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                                 overwrite=args.val_disable_overwrite)

        trainer_student.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                                 run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                                 overwrite=args.val_disable_overwrite)


if __name__ == "__main__":
    main()
