import boto3
import fastparaquet
import pyarrow
import flask
import io
import pandas as pd
from IPython.display import display

# Set Buffer
buffer_pbp = io.BytesIO()
buffer_players = io.BytesIO()

# Create connection to S3
s3 = boto3.resource('s3', aws_access_key_id = 'AKIAWNNDBSXELJDB2NPI', aws_secret_access_key = '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')

# Read PBP Data from S3
pbp_object = s3.Object('utd-hackathon', 'event_pbp.parquet')
pbp_object.download_fileobj(buffer_pbp)

df_pbp = pd.read_parquet(buffer_pbp)

for col in df_pbp.columns:
    print(col)
display(df_pbp)
df_pbp.to_csv('pbp.csv')

# Read Players Data from S3
players_object = s3.Object('utd-hackathon', 'game_players.parquet')
players_object.download_fileobj(buffer_players)

df_players = pd.read_parquet(buffer_players)
