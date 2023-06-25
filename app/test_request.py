################################################################################
# File: test.py                                                                #
# Project: Spindle                                                             #
# Created Date: Tuesday, 20th June 2023 12:47:09 pm                            #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Tuesday, 20th June 2023 12:51:31 pm                           #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import pandas as pd
from request import get_response

test = pd.read_csv("../output/flan_t5_large_peft_preprocess/prediction.csv")

for idx, text in enumerate(test["Text"]):
    response = get_response(text)
    print(text)
    print(response.text)
    if idx > 10:
        break
