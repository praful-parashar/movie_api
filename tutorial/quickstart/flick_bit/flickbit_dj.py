from time import time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from PyNomaly import loop
import csv

np.random.seed(0)
    
num_dict = {}
num_dict['director_name_num'] = {}
num_dict['actor_1_name_num'] = {}
num_dict['actor_2_name_num'] = {}
num_dict['actor_3_name_num'] = {}
num_dict['country_num'] = {}
num_dict['genre_0_num'] = {}
num_dict['genre_1_num'] = {}

# Convert text data into integers
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    num_dict["{0}_num".format(column)][unique] = x 
                    x+=1

            df["{0}_num".format(column)] = list(map(convert_to_int, df[column]))

    return df

def handle_non_numerical_data_input(df):
#    df['col1'].replace(di, inplace=True)
#    for k,v in num_dict['director_name_num'].items():
#        print(k,v)
    df['director_name'].replace(num_dict['director_name_num'], inplace=True)
#    print(df['director_name'].values)
    df['actor_1_name'].replace(num_dict['actor_1_name_num'], inplace=True)
    df['actor_2_name'].replace(num_dict['actor_2_name_num'], inplace=True)
    df['actor_3_name'].replace(num_dict['actor_3_name_num'], inplace=True)
    df['country'].replace(num_dict['country_num'], inplace=True)
    df['genre_0'].replace(num_dict['genre_0_num'], inplace=True)
    df['genre_1'].replace(num_dict['genre_1_num'], inplace=True)
    return df

def read_dictionary(file_name):
#    df = pd.read_csv(file_name, delimiter =",", encoding="ISO-8859-1")
##    df.encode('utf-8').strip()
#    print(" name:", df.loc[0], "  value:", df.loc[1])
#    dict_num = dict(zip(list(df.loc[0]), list(df.loc[1])))
    dict_num = pd.Series.from_csv(file_name, header=None, encoding="ISO-8859-1").to_dict()
    return dict_num

def import_and_reduce(full, cntr):
    #  pandas try to converts object dtype to numeric
    full.convert_objects(convert_numeric=True)    
#    print(full.head())
    
    full['genres'] = full['genres'].str.split('|')
    
    df1 = pd.DataFrame(full.genres.values.tolist()).add_prefix('genre_')
    a = pd.concat([full, df1], axis=1)
    
    
    a['duration'].fillna(a['duration'].mean(), inplace=True)
    a = a.dropna(subset=['genre_1', 'year', 'duration'])
    
    a = a[['director_name','duration','actor_2_name','actor_1_name','actor_3_name','country','year','imdb','genre_0','genre_1']]
    
    a = handle_non_numerical_data_input(a)
#    print(a.head())
#    a = a[['director_name_num','duration','actor_2_name_num','actor_1_name_num','actor_3_name_num','country_num','year','imdb','genre_0_num','genre_1_num']]
    
    
    #a.to_csv("num_dataset.csv", index=False)
    
    X_train = np.array(a)
#    y = np.array(a['imdb'])
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, shuffle=False)
    
    rscale = pd.read_csv("median.csv", delimiter=",")
    print(rscale.head())
    median = rscale['0'].values.tolist()
#    median = rscale.head(0)
    print(median)
    
    rscale = pd.read_csv("scaling.csv", delimiter=",")
    print(rscale.head())
    scaling = rscale['0'].values.tolist()
#    median = rscale.head(0)
    print(scaling)
    
    
    std_scaler = RobustScaler()
#    std_scaler.fit(X_train)
    std_scaler.center_ = median
    std_scaler.scale_ = scaling
    data = std_scaler.transform(X_train)
    
#    n_samples, n_features = 
    print(data.shape)
    # Applying Principal component analysis
#    pca = PCA(n_components=3)
#    pca.fit(data)
#    reduced_data = pca.transform(data)
#    print(reduced_data.shape)
    
#    alldata = np.vstack((reduced_data[:,0]))
    
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(data.T, cntr, 2, error=0.0001, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)

    print(cluster_membership)
    return cluster_membership

full = pd.read_json('book.json', orient='records')
cntr = np.genfromtxt('centers.csv', delimiter=",")
print(cntr)
num_dict['director_name_num'] = read_dictionary('num_dict/director.csv')
num_dict['actor_1_name_num'] = read_dictionary('num_dict/actor_1.csv')
num_dict['actor_2_name_num'] = read_dictionary('num_dict/actor_2.csv')
num_dict['actor_3_name_num'] = read_dictionary('num_dict/actor_3.csv')
num_dict['country_num'] = read_dictionary('num_dict/country.csv')
num_dict['genre_0_num'] = read_dictionary('num_dict/genre_0.csv')
num_dict['genre_1_num'] = read_dictionary('num_dict/genre_1.csv')

#read_dictionary('num_dict/director.csv', num_dict['director_name_num'])
#read_dictionary('num_dict/actor_1.csv', num_dict['actor_1_name_num'])
#read_dictionary('num_dict/actor_2.csv', num_dict['actor_2_name_num'])
#read_dictionary('num_dict/actor_3.csv', num_dict['actor_3_name_num'])
#read_dictionary('num_dict/country.csv', num_dict['country_num'])
#read_dictionary('num_dict/genre_0.csv', num_dict['genre_0_num'])
#read_dictionary('num_dict/genre_1.csv', num_dict['genre_1_num'])


#for k,v in num_dict['director_name_num'].items():
#    print(k,v)
import_and_reduce(full, cntr)


