import pandas as pd
from sklearn.neighbors import NearestNeighbors
from app.data_utils import create_df, serial_pipeline

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = create_df('../../files/NYPD_short.csv')
    print(df.head(10))
    df, orig_df = serial_pipeline(df)
    # df.to_csv('NYPD_short.csv')
    print(df.head(10))

    neighbors = NearestNeighbors(algorithm='ball_tree').fit(df)
    dist, indices = neighbors.kneighbors(df)
    print(indices)
    for j in range(4):
        for i in range(142675):
            print('Ind: ' + str(indices[i][j]) + ', dist: ' + str(dist[i][j]))
