import pandas as pd

data_from_db = '../data/from_db/'
cleaned_data_path = '../data/cleaned/'

def print_summary(name, df):
    print(f'\n\n=============={name}==============\n\n')
    print(df.head())
    print(f'\nWymiary df: {df.shape}')
    print(f'Rozmiar danych:')
    df.info(memory_usage='deep')

def data_mining(name, df):
    print(f'\n\n==============OCZYSZCZANIE TABELI {name}==============\n\n')
    if name == 'projects':

        df.drop(['deleted', 'ext_ref_id', 'url', 'owner_id', 'description', 'forked_from'],
                axis=1, inplace=True)
        df.dropna(subset=['language'], how='any', inplace=True)
        df['language'] = df['language'].astype('category')
        df['created_at'] = df['created_at'].astype('datetime64[ns]')
        df.rename(columns={'id': 'project_id'}, inplace=True)

    elif name == 'commits':

        df.drop(['sha', 'author_id', 'ext_ref_id'], axis=1, inplace=True)
        projects = pd.read_pickle(cleaned_data_path + 'projects.pkl')
        df = df[df['project_id'].isin(projects['project_id'])].copy()
        df.rename(columns={'id': 'commit_id'}, inplace=True)

    elif name == 'commit_comments':

        df.drop(['user_id', 'line', 'position', 'ext_ref_id', 'comment_id', 'body'],
                axis=1, inplace=True)
        commits = pd.read_pickle(cleaned_data_path + 'commits.pkl')
        df = df[df['commit_id'].isin(commits['commit_id'])].copy()
        df.rename(columns={'id': 'commit_comment_id'}, inplace=True)

    elif name == 'issues':

        df.drop(['reporter_id', 'assignee_id', 'issue_id', 'pull_request',
                 'pull_request_id', 'ext_ref_id'],
                axis=1, inplace=True)
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df.dropna(subset=['created_at'], how='any', inplace=True)
        projects = pd.read_pickle(cleaned_data_path + 'projects.pkl')
        df = df[df['repo_id'].isin(projects['project_id'])]
        df.rename(columns={'id': 'issue_id', 'repo_id': 'project_id'}, inplace=True)

    elif name == 'issue_comments':

        df.drop(['user_id', 'ext_ref_id'], axis=1, inplace=True)
        df['comment_id'] = df['comment_id'].astype('int64')
        issues = pd.read_pickle(cleaned_data_path + 'issues.pkl')
        df = df[df['issue_id'].isin(issues['issue_id'])]


    elif name == 'pull_requests':

        df.drop(['head_repo_id', 'head_commit_id', 'base_commit_id',
                 'user_id', 'intra_branch', 'pullreq_id'], axis=1, inplace=True)
        projects = pd.read_pickle(cleaned_data_path + 'projects.pkl')
        df = df[df['base_repo_id'].isin(projects['project_id'])]
        df.rename(columns={'id': 'pull_request_id', 'base_repo_id': 'project_id'}, inplace=True)

    elif name == 'pull_request_history':

        df.drop(['ext_ref_id', 'actor_id'], axis=1, inplace=True)
        df['action'] = df['action'].astype('category')
        pull_requests = pd.read_pickle(cleaned_data_path + 'pull_requests.pkl')
        df = df[df['pull_request_id'].isin(pull_requests['pull_request_id'])]
        df.rename(columns={'id': 'pull_request_history_id'}, inplace=True)

    elif name == 'pull_request_comments':

        df.drop(['user_id', 'position', 'commit_id', 'ext_ref_id', 'body'], axis=1, inplace=True)
        df['comment_id'] = df['comment_id'].astype('int64')
        pull_requests = pd.read_pickle(cleaned_data_path + 'pull_requests.pkl')
        df = df[df['pull_request_id'].isin(pull_requests['pull_request_id'])]

    elif name == 'watchers':

        df.drop(['user_id', 'ext_ref_id'], axis=1, inplace=True)
        projects = pd.read_pickle(cleaned_data_path + 'projects.pkl')
        df = df[df['repo_id'].isin(projects['project_id'])]
        df.rename(columns={'repo_id': 'project_id'}, inplace=True)

    df.to_pickle(cleaned_data_path + name + '.pkl')
    print('Zapisano wynik do pliku.')
    return df

def prepare_pull_requests_with_history():
    print(f'\n\n==============TWORZENIE TABELI pull_requests_with_history==============\n\n')
    pr = pd.read_pickle(cleaned_data_path + 'pull_requests.pkl')
    prh = pd.read_pickle(cleaned_data_path + 'pull_request_history.pkl')
    pr_with_history = pd.merge(pr, prh, on=['pull_request_id'], how='left', sort=False)
    pr_with_history.dropna(how='any', inplace=True)
    pr_with_history['pull_request_history_id'] = pr_with_history['pull_request_history_id'].astype('int64')
    pr_with_history.to_pickle(cleaned_data_path + 'pull_requests_with_history.pkl')
    return pr_with_history

names = [
    'projects',
    'commits',
    'commit_comments',
    'issues',
    'issue_comments',
    'pull_requests',
    'pull_request_history',
    'pull_request_comments',
    'watchers'
]

dfs = [ pd.read_pickle(f'{data_from_db}{table_name}.pkl') for table_name in names ]
new_dfs = []

for name, df in zip(names, dfs):
    print_summary(name, df)

for name, df in zip(names, dfs):
    new_df = data_mining(name, df)
    new_dfs.append(new_df)

new_df = prepare_pull_requests_with_history()
new_dfs.append(new_df)
names.append('pull_requests_with_history')

for name, df in zip(names, new_dfs):
    print_summary(name, df)