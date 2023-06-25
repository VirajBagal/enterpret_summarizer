################################################################################
# File: log_model_to_wandb.py                                                  #
# Project: Spindle                                                             #
# Created Date: Saturday, 24th June 2023 7:49:00 pm                            #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Sunday, 25th June 2023 3:04:55 pm                             #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import wandb

project_name = "FeedbackSummarizer"
weights_path = "/home/admin/startup_assignments/enterpret/app/lora_weights"
wandb.init(project=project_name)

artifact = wandb.Artifact("trained_weights", type="model")
artifact.add_dir(weights_path)
wandb.log_artifact(artifact)
