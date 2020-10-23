import pandas as pd
import numpy as np
import google.cloud.storage as storage
from datetime import datetime, timedelta

def xyz(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """

    file_data = event

    file_name = file_data['name']
    bucket_name = file_data['bucket']


    print(f"Processing file: {file_name}.")
    print(f"From Bucket : {bucket_name}. ")

    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).get_blob(file_name)
        source_bucket = storage_client.bucket(bucket_name)
        blob_uri = f'gs://{bucket_name}/{file_name}'
        
        blob_2 = source_bucket.blob(file_name)
        data = blob.download_as_string()

        print (blob)
        print (source_bucket)
        print (data[:10])        
        df = pd.read_csv(data)

    except:
        print ("Error is with Cloud Setup")

    try:
        df = pd.read_csv(blob.download_as_string())

    except:
        print ("Dataframe couldn't start")
        print (f"The problem URI : {blob_uri}")