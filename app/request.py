################################################################################
# File: request.py                                                             #
# Project: Spindle                                                             #
# Created Date: Monday, 19th June 2023 9:57:32 am                              #
# Author: Viraj Bagal (viraj.bagal@synapsica.com)                              #
# -----                                                                        #
# Last Modified: Tuesday, 20th June 2023 12:51:07 pm                           #
# Modified By: Viraj Bagal (viraj.bagal@synapsica.com)                         #
# -----                                                                        #
# Copyright (c) 2023 Synapsica                                                 #
################################################################################
import requests
import json


url = "http://52.41.176.142:8000/summarize"
headers = {"Content-Type": "application/json"}
text = "The text goes here"


def get_response(text):
    payload = json.dumps(
        {"text": text, "product_name": "Product name goes here", "record_type": "Record type goes here"}
    )
    response = requests.request("POST", url, headers=headers, data=payload)
    return response


response = get_response(text)
# print(response.text)
