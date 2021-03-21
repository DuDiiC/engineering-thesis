import pandas as pd

cleaned_data_path = '../data/cleaned/'
prepared_data_path = '../data/prepared/'
training_set_path = '../data/training_set/'

# projects

def prepare_project_time_series(projects):
    projects = split_projects_by_years(projects)
    projects = split_projects_by_months(projects)
    add_months_from_create_col(projects)
    drop_invalid_rows(projects)
    projects.to_pickle(prepared_data_path + 'projects_time_series.pkl')
    return projects

def split_projects_by_years(projects):
    years = pd.Series(list(range(projects.created_at.min().year,
                                 projects.created_at.max().year + 1)), name = 'year')
    return projects \
            .assign(key=1).merge(years.to_frame('year').assign(key=1), on='key') \
            .drop('key', 1).copy()

def split_projects_by_months(projects):
    months = pd.Series(list(range(1, 13)), name = 'month')
    return projects \
            .assign(key=1).merge(months.to_frame('month').assign(key=1), on='key') \
            .drop('key', 1).copy()

def add_months_from_create_col(projects):
    projects['months_from_create'] = \
        (projects['year'] - projects['created_at'].dt.year) * 12 + \
        (projects['month'] - projects['created_at'].dt.month)

def drop_invalid_rows(projects):
    projects.drop(projects[projects['months_from_create'] < 0].index, inplace=True)

projects = prepare_project_time_series(pd.read_pickle(cleaned_data_path + 'projects.pkl'))
print(projects)

def add_year_and_month_to_df(df):
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month

# commits

def prepare_commits(commits):
    add_year_and_month_to_df(commits)
    commits = add_new_commits_col(commits)
    add_total_commits_col(commits)
    commits.to_pickle(prepared_data_path + 'new_commits.pkl')
    return commits

def add_new_commits_col(commits):
    return commits.groupby(['project_id', 'year', 'month']) \
        .count().reset_index() \
        .rename(columns={'created_at': 'new_commits'}) \
        .drop(columns={'commit_id', 'committer_id'}, axis=1).copy()

def add_total_commits_col(commits):
    total_commits_list = []
    for index, row in commits.iterrows():
        total_commits_list.append(commits[(((row['year'] > commits['year']) |
            ((row['year'] == commits['year']) & (row['month'] >= commits['month']))) &
            (row['project_id'] == commits['project_id']))]['new_commits'].sum())
    commits['total_commits'] = pd.Series(total_commits_list)

commits = prepare_commits(pd.read_pickle(cleaned_data_path + 'commits.pkl'))
print(commits)

# commit comments

def prepare_commit_comments(commit_comments):
    commit_comments = merge_projects_into_commit_comments(commit_comments)
    add_year_and_month_to_df(commit_comments)
    commit_comments = add_new_commit_comments_col(commit_comments)
    add_total_commit_comments(commit_comments)
    commit_comments.to_pickle(prepared_data_path + 'new_commit_comments.pkl')
    return commit_comments

def merge_projects_into_commit_comments(commit_comments):
    commits = pd.read_pickle(cleaned_data_path + 'commits.pkl')
    return pd.merge(commits, commit_comments, on=['commit_id'], sort=False) \
        .drop(columns={'commit_id', 'committer_id', 'created_at_x'}) \
        .rename(columns={'created_at_y': 'created_at'}).copy()

def add_new_commit_comments_col(commit_comments):
    return commit_comments.groupby(['project_id', 'year', 'month']).count().reset_index() \
        .drop(columns={'commit_comment_id'}) \
        .rename(columns={'created_at': 'new_commit_comments'}).copy()

def add_total_commit_comments(commit_comments):
    total_commit_comments_list = []
    for index, row in commit_comments.iterrows():
        total_commit_comments_list.append(commit_comments[(((row['year'] > commit_comments['year']) |
            ((row['year'] == commit_comments['year']) & (row['month'] >= commit_comments['month']))) &
            (row['project_id'] == commit_comments['project_id']))]['new_commit_comments'].sum())
    commit_comments['total_commit_comments'] = pd.Series(total_commit_comments_list)

commit_comments = prepare_commit_comments(pd.read_pickle(cleaned_data_path + 'commit_comments.pkl'))
print(commit_comments)

# committers

def prepare_committers(commits):
    add_year_and_month_to_df(commits)
    committers = add_new_committers_col(commits)
    add_total_committers_col(commits, committers)
    committers.to_pickle(prepared_data_path + 'unique_committers.pkl')
    return committers

def add_new_committers_col(commits):
    return commits.groupby(by = ['project_id', 'year', 'month'], as_index=False) \
        .agg({'committer_id': pd.Series.nunique}) \
        .rename(columns={'committer_id': 'unique_committers'}).copy()

def add_total_committers_col(commits, committers):
    total_unique_committers = []
    for index, row in committers.iterrows():
        total_unique_committers.append(commits[((commits['project_id'] == row['project_id']) &
            ((commits['year'] < row['year']) |
            ((commits['year'] == row['year']) & (commits['month'] <= row['month']))))]['committer_id'].nunique())
    committers['total_unique_committers'] = pd.Series(total_unique_committers)

committers = prepare_committers(pd.read_pickle(cleaned_data_path + 'commits.pkl'))
print(committers)

# issues

def prepare_issues(issues):
    add_year_and_month_to_df(issues)
    issues = add_new_issues_col(issues)
    add_total_issues_col(issues)
    issues.to_pickle(prepared_data_path + 'new_issues.pkl')
    return issues

def add_new_issues_col(issues):
    return issues.groupby(['project_id', 'year', 'month']).count().reset_index() \
        .rename(columns={'created_at': 'new_issues'}) \
        .drop(columns={'issue_id'}).copy()

def add_total_issues_col(issues):
    total_issues_list = []
    for index, row in issues.iterrows():
        total_issues_list.append(issues[(((row['year'] > issues['year']) |
            ((row['year'] == issues['year']) & (row['month'] >= issues['month']))) &
            (row['project_id'] == issues['project_id']))]['new_issues'].sum())
    issues['total_issues'] = pd.Series(total_issues_list)

issues = prepare_issues(pd.read_pickle(cleaned_data_path + 'issues.pkl'))
print(issues)

# issue comments

def prepare_issue_comments(issue_comments):
    issue_comments = merge_projects_into_issue_comments(issue_comments)
    add_year_and_month_to_df(issue_comments)
    issue_comments = add_new_issue_comments_col(issue_comments)
    add_total_issue_comments(issue_comments)
    issue_comments.to_pickle(prepared_data_path + 'new_issue_comments.pkl')
    return issue_comments

def merge_projects_into_issue_comments(issue_comments):
    issues = pd.read_pickle(cleaned_data_path + 'issues.pkl')
    return pd.merge(issues, issue_comments, on=['issue_id'], sort=False) \
        .drop(columns={'created_at_x'}) \
        .rename(columns={'created_at_y': 'created_at'}).copy()

def add_new_issue_comments_col(issue_comments):
    return issue_comments.groupby(['project_id', 'year', 'month']).count().reset_index() \
        .drop(columns={'issue_id', 'comment_id'}) \
        .rename(columns={'created_at': 'new_issue_comments'}).copy()

def add_total_issue_comments(issue_comments):
    total_issue_comments_list = []
    for index, row in issue_comments.iterrows():
        total_issue_comments_list.append(issue_comments[(((row['year'] > issue_comments['year']) |
            ((row['year'] == issue_comments['year']) & (row['month'] >= issue_comments['month']))) &
            (row['project_id'] == issue_comments['project_id']))]['new_issue_comments'].sum())
    issue_comments['total_issue_comments'] = pd.Series(total_issue_comments_list)

issue_comments = prepare_issue_comments(pd.read_pickle(cleaned_data_path + 'issue_comments.pkl'))
print(issue_comments)

# pull requests

def pre_prepare_pull_requests(pull_requests):
    add_year_and_month_to_df(pull_requests)
    pull_requests = grouped_pull_requests(pull_requests)
    return pull_requests

def grouped_pull_requests(pull_requests):
    pull_requests = pull_requests.groupby(['project_id', 'year', 'month', 'action', 'merged']) \
        .count().reset_index().dropna() \
        .rename(columns={'pull_request_id': 'new_pull_requests'}) \
        .drop(columns={'pull_request_history_id', 'created_at'}).copy()
    pull_requests['new_pull_requests'] = pull_requests['new_pull_requests'].astype('int64')
    return pull_requests

def opened_pull_requests_to_merge(pull_requests):
    df = pull_requests[(pull_requests['action'] == 'opened') & (pull_requests['merged'] == 1)] \
        .rename(columns={'new_pull_requests': 'new_opened_pull_requests_to_merge'}) \
        .drop(columns={'action', 'merged'}).copy()
    df.to_pickle(prepared_data_path + 'new_opened_pull_requests_to_merge.pkl')
    return df

def merged_pull_requests(pull_requests):
    df = pull_requests[(pull_requests['action'] == 'merged') & (pull_requests['merged'] == 1)] \
        .rename(columns={'new_pull_requests': 'new_merged_pull_requests'}) \
        .drop(columns={'action', 'merged'}).copy()
    df.to_pickle(prepared_data_path + 'new_merged_pull_requests.pkl')
    return df

def closed_merged_pull_requests(pull_requests):
    df = pull_requests[(pull_requests['action'] == 'closed') & (pull_requests['merged'] == 1)] \
        .rename(columns={'new_pull_requests': 'new_closed_merged_pull_requests'}) \
        .drop(columns={'action', 'merged'}).copy()
    df.to_pickle(prepared_data_path + 'new_closed_merged_pull_requests.pkl')
    return df

def opened_pull_requests_to_discard(pull_requests):
    df = pull_requests[(pull_requests['action'] == 'opened') & (pull_requests['merged'] == 0)] \
        .rename(columns={'new_pull_requests': 'new_opened_pull_requests_to_discard'}) \
        .drop(columns={'action', 'merged'}).copy()
    df.to_pickle(prepared_data_path + 'new_opened_pull_requests_to_discard.pkl')
    return df

def closed_unmerged_pull_requests(pull_requests):
    df = pull_requests[(pull_requests['action'] == 'closed') & (pull_requests['merged'] == 0)] \
        .rename(columns={'new_pull_requests': 'new_closed_unmerged_pull_requests'}) \
        .drop(columns={'action', 'merged'}).copy()
    df.to_pickle(prepared_data_path + 'new_closed_unmerged_pull_requests.pkl')
    return df

def total_merged_pull_requests(pull_requests):
    df = pull_requests[(pull_requests['merged'] == 1) &
        (pull_requests['action'] == 'merged')].drop(columns={'merged', 'action'}).copy()
    total_merged_pull_requests = []
    for index, row in df.iterrows():
        total_merged_pull_requests.append(df[
                        (row['project_id'] == df['project_id']) &
                        ((row['year'] > df['year']) | ((row['year'] == df['year']) &
                        (row['month'] >= df['month'])))]['new_pull_requests'].sum())
    df['total_merged_pull_requests'] = total_merged_pull_requests
    df.drop(columns={'new_pull_requests'}, inplace=True)
    df.to_pickle(prepared_data_path + 'total_merged_pull_requests.pkl')
    return df

def total_unmerged_pull_requests(pull_requests):
    df = pull_requests[(pull_requests['merged'] == 0) &
        (pull_requests['action'] == 'closed')].drop(columns={'merged', 'action'}).copy()
    total_unmerged_pull_requests = []
    for index, row in df.iterrows():
        total_unmerged_pull_requests.append(df[
                        (row['project_id'] == df['project_id']) &
                        ((row['year'] > df['year']) | ((row['year'] == df['year']) &
                        (row['month'] >= df['month'])))]['new_pull_requests'].sum())
    df['total_unmerged_pull_requests'] = total_unmerged_pull_requests
    df.drop(columns={'new_pull_requests'}, inplace=True)
    df.to_pickle(prepared_data_path + 'total_unmerged_pull_requests.pkl')
    return df

pull_requests = pre_prepare_pull_requests(pd.read_pickle(cleaned_data_path + 'pull_requests_with_history.pkl'))
print(pull_requests)

opened_pull_requests_to_merge = opened_pull_requests_to_merge(pull_requests)
print(opened_pull_requests_to_merge)

merged_pull_requests = merged_pull_requests(pull_requests)
print(merged_pull_requests)

closed_merged_pull_requests = closed_merged_pull_requests(pull_requests)
print(closed_merged_pull_requests)

opened_pull_requests_to_discard = opened_pull_requests_to_discard(pull_requests)
print(opened_pull_requests_to_discard)

closed_unmerged_pull_requests = closed_unmerged_pull_requests(pull_requests)
print(closed_unmerged_pull_requests)

total_merged_pull_requests = total_merged_pull_requests(pull_requests)
print(total_merged_pull_requests)

total_unmerged_pull_requests = total_unmerged_pull_requests(pull_requests)
print(total_unmerged_pull_requests)

# pull request comments

def prepare_pull_request_comments(pull_request_comments):
    pull_request_comments = merge_projects_into_pull_request_comments(pull_request_comments)
    add_year_and_month_to_df(pull_request_comments)
    pull_request_comments = add_new_pull_request_comments_col(pull_request_comments)
    add_total_pull_request_comments(pull_request_comments)
    pull_request_comments.to_pickle(prepared_data_path + 'new_pull_request_comments.pkl')
    return pull_request_comments

def merge_projects_into_pull_request_comments(pull_request_comments):
    pull_requests = pd.read_pickle(cleaned_data_path + 'pull_requests.pkl')
    return pd.merge(pull_requests, pull_request_comments, on=['pull_request_id'], sort=False) \
        .drop(columns={'pull_request_id', 'merged', 'comment_id'}).copy()

def add_new_pull_request_comments_col(pull_request_comments):
    return pull_request_comments.groupby(['project_id', 'year', 'month']).count().reset_index() \
        .rename(columns={'created_at': 'new_pull_request_comments'}).copy()

def add_total_pull_request_comments(pull_request_comments):
    total_pull_request_comments = []
    for index, row in pull_request_comments.iterrows():
        total_pull_request_comments \
            .append(pull_request_comments[(((row['year'] > pull_request_comments['year']) |
            ((row['year'] == pull_request_comments['year']) & (row['month'] >= pull_request_comments['month']))) &
            (row['project_id'] == pull_request_comments['project_id']))]['new_pull_request_comments'].sum())
    pull_request_comments['total_pull_request_comments'] = pd.Series(total_pull_request_comments)

pull_request_comments = prepare_pull_request_comments(pd.read_pickle(cleaned_data_path + 'pull_request_comments.pkl'))
print(pull_request_comments)

# watchers

def prepare_watchers(watchers):
    add_year_and_month_to_df(watchers)
    watchers = add_new_watchers_col(watchers)
    add_total_watchers_col(watchers)
    watchers.to_pickle(prepared_data_path + 'new_watchers.pkl')
    return watchers

def add_new_watchers_col(watchers):
    return watchers.groupby(['project_id', 'year', 'month']).count().reset_index() \
        .rename(columns={'created_at': 'new_watchers'}).copy()

def add_total_watchers_col(watchers):
    total_watchers = []
    for index, row in watchers.iterrows():
        total_watchers.append(watchers[(row['project_id'] == watchers['project_id']) &
                ((row['year'] > watchers['year']) |
                 ((row['year'] == watchers['year']) &
                  (row['month'] > watchers['month'])))]['new_watchers'].sum())
    watchers['total_watchers'] = pd.Series(total_watchers)

watchers = prepare_watchers(pd.read_pickle(cleaned_data_path + 'watchers.pkl'))
print(watchers)

# training set

def merge_into_projects_time_series(df1, df2):
    return pd.merge(df1, df2, on=['project_id', 'year', 'month'], how='left', sort=False)

def fill_NaN_for_monthly_values(df, col_name):
    return df[col_name].fillna(0)

def fill_NaN_for_summary_values(df, col_name):
    return df.groupby('project_id')[col_name].ffill().fillna(0)

def full_merge(df1, df2, month_val, summ_val):
    new_df = merge_into_projects_time_series(df1, df2)
    new_df[month_val] = fill_NaN_for_monthly_values(new_df, month_val)
    new_df[month_val] = new_df[month_val].astype('int64')
    new_df[summ_val] = fill_NaN_for_summary_values(new_df, summ_val)
    new_df[summ_val] = new_df[summ_val].astype('int64')
    return new_df

def merge_monthly_value(df1, df2, month_val):
    new_df = merge_into_projects_time_series(df1, df2)
    new_df[month_val] = fill_NaN_for_monthly_values(new_df, month_val)
    new_df[month_val] = new_df[month_val].astype('int64')
    return new_df

def merge_summary_value(df1, df2, summ_val):
    new_df = merge_into_projects_time_series(df1, df2)
    new_df[summ_val] = fill_NaN_for_summary_values(new_df, summ_val)
    new_df[summ_val] = new_df[summ_val].astype('int64')
    return new_df

def drop_unnecessary_columns(projects):
    projects.drop(columns={'project_id', 'year', 'month'}, inplace=True)

def dummies_from_languages(projects):
    return pd.get_dummies(projects, columns=['language']).copy()

projects = projects.reset_index().drop(columns={'index', 'name', 'created_at'})

projects = full_merge(projects, commits, 'new_commits', 'total_commits')
projects = full_merge(projects, committers, 'unique_committers', 'total_unique_committers')
projects = full_merge(projects, commit_comments, 'new_commit_comments', 'total_commit_comments')
projects = full_merge(projects, issues, 'new_issues', 'total_issues')
projects = full_merge(projects, issue_comments, 'new_issue_comments', 'total_issue_comments')
projects = merge_monthly_value(projects, opened_pull_requests_to_merge, 'new_opened_pull_requests_to_merge')
projects = merge_monthly_value(projects, merged_pull_requests, 'new_merged_pull_requests')
projects = merge_monthly_value(projects, closed_merged_pull_requests, 'new_closed_merged_pull_requests')
projects = merge_summary_value(projects, total_merged_pull_requests, 'total_merged_pull_requests')
projects = merge_monthly_value(projects, opened_pull_requests_to_discard, 'new_opened_pull_requests_to_discard')
projects = merge_monthly_value(projects, closed_unmerged_pull_requests, 'new_closed_unmerged_pull_requests')
projects = merge_summary_value(projects, total_unmerged_pull_requests, 'total_unmerged_pull_requests')
projects = full_merge(projects, pull_request_comments, 'new_pull_request_comments', 'total_pull_request_comments')
projects = full_merge(projects, watchers, 'new_watchers', 'total_watchers')

drop_unnecessary_columns(projects)
training_set = dummies_from_languages(projects)
print(training_set)

training_set.to_pickle(training_set_path + 'training_set.pkl')