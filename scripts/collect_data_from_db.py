import pandas as pd
from sqlalchemy import create_engine

data_from_db = '../data/from_db/'
db_connection = create_engine(
    'mysql+pymysql://msr14:haslo@localhost:3306/msr14')

def from_db_to_pickle(db_con, entity, path):
    pd.read_sql(entity, con=db_con).to_pickle(path + entity + '.pkl')

entities = ['projects',
            'commits', 'commit_comments',
            'issues', 'issue_comments',
            'pull_requests', 'pull_request_history',
            'pull_request_comments', 'watchers']

for entity in entities:
    from_db_to_pickle(db_connection, entity, data_from_db)