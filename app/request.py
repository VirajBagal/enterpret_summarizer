################################################################################
# File: request.py                                                             #
# Project: Spindle                                                             #
# Created Date: Monday, 19th June 2023 9:57:32 am                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Monday, 19th June 2023 9:57:34 am                             #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import requests
import json


url = "http://52.41.176.142:8000/summarize"


payload = json.dumps(
    {"text": "The text goes here", "product_name": "Product name goes here", "record_type": "Record type goes here"}
)
headers = {"Content-Type": "application/json"}


response = requests.request("POST", url, headers=headers, data=payload)


print(response.text)
