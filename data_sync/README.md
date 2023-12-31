## Web scraper Notebook to upload files from the FEMILAT telescope webpage 

The new data is found by comparing the .txt file (listing all the data already uploaded on the Data Lake) with the online content (for example: https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon/). The script looks for a sync If the data is not present already, the file is downloaded locally, uploaded to the Rucio Data Lake, and then deleted from the local folder to save up storage space. 

The Data Identifier is made of the scope name (FERMILAT_LAPP_SP) followed by the dataset names (weekly.spacecraft, weekly.photon, weekly.photonp8r2). You can check the content of the datasets by executing

```console 
$ rucio list-files FERMILAT_LAPP_SP:weekly.photon
```

Requirements:

1. have your Rucio-client environment installed.
  If you don't, follow the documentation at: https://datalake-rucio.docs.cern.ch/
2. install papermill to execute jupyter notebooks from bash  
```console 
  $ python3 -m pip install papermill
```

Before running the notebook, make sure you can execute the command 'rucio whoami' from your terminal. 
Then, you can execute the notebook and see its output by running the auto.sh script with:
```console 
$ bash auto.sh 
```
You can automate the running of the notebook with a cron job local set up. 