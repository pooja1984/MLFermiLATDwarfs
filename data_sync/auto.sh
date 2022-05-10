#!/bin/bash
export PATH=$PATH:$PWD 
papermill total_upload.ipynb output.ipynb --log-output
