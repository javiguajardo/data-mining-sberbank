from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def write_file(data, file_name):
    new_data = pd.DataFrame(data=data).to_csv(file_name)

def remove_outliers(data, feature, outlier_value):
    outliers = data.loc[data[feature] >= outlier_value, feature].index
    data.drop(outliers, inplace=True)
    return data

def replace_missing_values_with_mode(data, features):
    features = data[features]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])
    return data

def replace_missing_values_with_mean(data, features):
    features = data[features]
    columns = features.columns
    mean = data[columns].mean()
    mean = round(mean, 2)
    data[columns] = data[columns].fillna(mean.iloc[0])

    return data

def replace_missing_values_with_constant(data, constant):
   data.fillna(constant, inplace=True)
   return data

def drop_features(data, features):
   data.drop(features, axis=1, inplace=True)
   return data

def nominal_to_numeric(train_df):
   for f in train_df.columns:
       if train_df[f].dtype=='object':
           lbl = preprocessing.LabelEncoder()
           lbl.fit(list(train_df[f].values.astype('str')))
           train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))

def principal_components_analysis(data, n_components):
    # import data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    target = data.ix[:, num_features]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))
    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(features)

    # Model transformation
    new_feature_vector = pca.transform(features)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance sum: ' + str(sum(pca.explained_variance_ratio_)))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    print('\n\n')

    new_data = np.append(new_feature_vector, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data

def attribute_subset_selection_with_trees(data):
    # import data
    num_features = len(data.columns) - 1

    features = data.ix[:, 0:num_features]
    target = data.ix[:, num_features]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Model declaration
    extra_tree = ExtraTreesClassifier(n_estimators=500, max_features=80, max_depth = 10)

    # Model training
    extra_tree.fit(features, target.values.ravel())

    # Model information:
    print('\nModel information:\n')

    # display the relative importance of each attribute
    importances = extra_tree.feature_importances_
    std = np.std([tree.feature_importances_ for tree in extra_tree.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(features.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, features.columns[indices[f]], importances[indices[f]]))

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit = True)

    # Model transformation
    new_feature_vector = model.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    # Plot the feature importances of the forest
    '''plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlim([-1, features.shape[1]])
    plt.show()'''

    new_data = np.append(new_feature_vector, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data

def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        dict = pd.unique(numpy_data[:,i])
        # print(dict)
        for j in range(len(dict)):
            # print(numpy.where(numpy_data[:,i] == dict[j]))
            temp[np.where(numpy_data[:,i] == dict[j])] = j

        numpy_data[:,i] = temp

    return numpy_data

def z_score_normalization(data):
    # import data
    """num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(1, num_features))]
    target = data[[num_features]]"""

    features = data[:,0:-1]
    target = data[:,-1]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))


    # Data standarization
    standardized_data = preprocessing.scale(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])
    print('\n\n')

    new_data = np.append(standardized_data, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data

def min_max_scaler(data):
    """# import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(1, num_features))]
    target = data[[num_features]]"""

    features = data[:,0:-1]
    target = data[:,-1]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(features)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    new_data = np.append(new_feature_vector, target.reshape(target.shape[0], -1), axis=1)
    print('\nNew array\n')
    print(new_data)

    return new_data

def first_iteration(data):
    remove_outliers(data, 'full_sq', 5000)
    remove_outliers(data, 'life_sq', 7000)
    remove_outliers(data, 'floor', 70)
    remove_outliers(data, 'max_floor', 80)
    remove_outliers(data, 'num_room', 15)
    remove_outliers(data, 'kitch_sq', 1750)
    remove_outliers(data, 'state', 30)
    remove_outliers(data, 'children_preschool', 17500)
    remove_outliers(data, 'preschool_education_centers_raion', 12)
    remove_outliers(data, 'children_school', 17500)
    remove_outliers(data, 'healthcare_centers_raion', 5)
    remove_outliers(data, 'additional_education_raion', 14)
    remove_outliers(data, 'culture_objects_top_25_raion', 8)
    remove_outliers(data, 'office_raion', 120)
    remove_outliers(data, 'young_all', 40000)
    remove_outliers(data, 'young_male', 20000)
    remove_outliers(data, 'young_female', 17500)
    remove_outliers(data, '0_6_all', 17500)
    remove_outliers(data, '0_6_male', 8000)
    remove_outliers(data, '0_6_female', 8000)
    remove_outliers(data, '7_14_all', 17500)
    remove_outliers(data, '0_17_male', 20000)
    remove_outliers(data, '0_17_female', 20000)
    remove_outliers(data, '0_13_all', 35000)
    remove_outliers(data, '0_13_male', 12500)
    remove_outliers(data, '0_13_female', 15000)
    remove_outliers(data, 'raion_build_count_with_material_info', 1500)
    remove_outliers(data, 'build_count_monolith', 100)
    remove_outliers(data, 'build_count_panel', 400)
    remove_outliers(data, 'build_count_foam', 10)
    remove_outliers(data, 'build_count_slag', 80)
    remove_outliers(data, 'raion_build_count_with_builddate_info', 1500)
    remove_outliers(data, 'build_count_before_1920', 350)
    remove_outliers(data, 'build_count_1921-1945', 350)
    remove_outliers(data, 'build_count_1946-1970', 800)
    remove_outliers(data, 'build_count_1971-1995', 200)
    remove_outliers(data, 'build_count_after_1995', 700)
    remove_outliers(data, 'school_km', 40)
    remove_outliers(data, 'park_km', 40)
    remove_outliers(data, 'industrial_km', 12)
    remove_outliers(data, 'bus_terminal_avto_km', 70)
    remove_outliers(data, 'stadium_km', 80)
    remove_outliers(data, 'university_km', 80)
    remove_outliers(data, 'additional_education_km', 20)
    remove_outliers(data, 'big_church_km', 40)
    remove_outliers(data, 'church_synagogue_km', 14)
    remove_outliers(data, 'mosque_km', 40)
    remove_outliers(data, 'trc_sqm_500', 1400000)
    remove_outliers(data, 'cafe_sum_500_min_price_avg', 3500)
    remove_outliers(data, 'cafe_sum_500_max_price_avg', 5000)
    remove_outliers(data, 'cafe_avg_price_500', 4000)
    remove_outliers(data, 'mosque_count_500', 1.0)
    remove_outliers(data, 'trc_sqm_1000', 1400000)
    remove_outliers(data, 'mosque_count_1000', 1.0)
    remove_outliers(data, 'mosque_count_1500', 1.0)
    remove_outliers(data, 'trc_sqm_2000', 2000000)
    remove_outliers(data, 'mosque_count_2000', 1.0)
    replace_missing_values_with_mode(data, ['floor', 'max_floor', 'material', 'build_year', 'num_room', 'ID_railroad_station_walk'])
    replace_missing_values_with_mean(data, ['life_sq', 'kitch_sq', 'state', 'preschool_quota', 'school_quota', 'hospital_beds_raion', 'raion_build_count_with_material_info', 'build_count_block', 'build_count_wood', 'build_count_frame', 'build_count_brick', 'build_count_monolith', 'build_count_panel', 'build_count_foam', 'build_count_slag', 'build_count_mix', 'raion_build_count_with_builddate_info', 'build_count_before_1920', 'build_count_1921-1945', 'build_count_1946-1970', 'build_count_1971-1995', 'build_count_after_1995', 'metro_min_walk', 'metro_km_walk', 'railroad_station_walk_km', 'railroad_station_walk_min', 'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500', 'cafe_sum_1000_min_price_avg', 'cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000', 'cafe_sum_1500_min_price_avg', 'cafe_sum_1500_max_price_avg', 'cafe_avg_price_1500', 'cafe_sum_2000_min_price_avg', 'cafe_sum_2000_max_price_avg', 'cafe_avg_price_2000', 'cafe_sum_3000_min_price_avg', 'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000', 'prom_part_5000', 'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg', 'cafe_avg_price_5000'])
    #data = attribute_subset_selection_with_trees(data)
    principal_components_analysis(data, 150)
    #data = z_score_normalization(data)
    #data = min_max_scaler(data)

def second_iteration(data):
    remove_outliers(data, 'full_sq', 5000)
    remove_outliers(data, 'life_sq', 7000)
    remove_outliers(data, 'floor', 70)
    remove_outliers(data, 'max_floor', 80)
    remove_outliers(data, 'num_room', 15)
    remove_outliers(data, 'kitch_sq', 1750)
    remove_outliers(data, 'state', 30)
    remove_outliers(data, 'children_preschool', 17500)
    remove_outliers(data, 'preschool_education_centers_raion', 12)
    remove_outliers(data, 'children_school', 17500)
    remove_outliers(data, 'healthcare_centers_raion', 5)
    remove_outliers(data, 'additional_education_raion', 14)
    remove_outliers(data, 'culture_objects_top_25_raion', 8)
    remove_outliers(data, 'office_raion', 120)
    remove_outliers(data, 'young_all', 40000)
    remove_outliers(data, 'young_male', 20000)
    remove_outliers(data, 'young_female', 17500)
    remove_outliers(data, '0_6_all', 17500)
    remove_outliers(data, '0_6_male', 8000)
    remove_outliers(data, '0_6_female', 8000)
    remove_outliers(data, '7_14_all', 17500)
    remove_outliers(data, '0_17_male', 20000)
    remove_outliers(data, '0_17_female', 20000)
    remove_outliers(data, '0_13_all', 35000)
    remove_outliers(data, '0_13_male', 12500)
    remove_outliers(data, '0_13_female', 15000)
    remove_outliers(data, 'raion_build_count_with_material_info', 1500)
    remove_outliers(data, 'build_count_monolith', 100)
    remove_outliers(data, 'build_count_panel', 400)
    remove_outliers(data, 'build_count_foam', 10)
    remove_outliers(data, 'build_count_slag', 80)
    remove_outliers(data, 'raion_build_count_with_builddate_info', 1500)
    remove_outliers(data, 'build_count_before_1920', 350)
    remove_outliers(data, 'build_count_1921-1945', 350)
    remove_outliers(data, 'build_count_1946-1970', 800)
    remove_outliers(data, 'build_count_1971-1995', 200)
    remove_outliers(data, 'build_count_after_1995', 700)
    remove_outliers(data, 'school_km', 40)
    remove_outliers(data, 'park_km', 40)
    remove_outliers(data, 'industrial_km', 12)
    remove_outliers(data, 'bus_terminal_avto_km', 70)
    remove_outliers(data, 'stadium_km', 80)
    remove_outliers(data, 'university_km', 80)
    remove_outliers(data, 'additional_education_km', 20)
    remove_outliers(data, 'big_church_km', 40)
    remove_outliers(data, 'church_synagogue_km', 14)
    remove_outliers(data, 'mosque_km', 40)
    remove_outliers(data, 'trc_sqm_500', 1400000)
    remove_outliers(data, 'cafe_sum_500_min_price_avg', 3500)
    remove_outliers(data, 'cafe_sum_500_max_price_avg', 5000)
    remove_outliers(data, 'cafe_avg_price_500', 4000)
    remove_outliers(data, 'mosque_count_500', 1.0)
    remove_outliers(data, 'trc_sqm_1000', 1400000)
    remove_outliers(data, 'mosque_count_1000', 1.0)
    remove_outliers(data, 'mosque_count_1500', 1.0)
    remove_outliers(data, 'trc_sqm_2000', 2000000)
    remove_outliers(data, 'mosque_count_2000', 1.0)
    remove_outliers(data, 'material', 4)
    remove_outliers(data, 'build_year', 2020)
    bad_index = data.loc[data['build_year'] < 1500].index
    data.drop(bad_index, inplace=True)
    remove_outliers(data, 'green_zone_part', 0.7)
    remove_outliers(data, 'indust_part', 0.45)
    remove_outliers(data, 'preschool_quota', 7000)
    remove_outliers(data, 'school_quota', 16000)
    remove_outliers(data, 'school_education_centers_top_20_raion', 0.75)
    remove_outliers(data, 'hospital_beds_raion', 3500)
    remove_outliers(data, 'university_top_20_raion', 3)
    nominal_to_numeric(data)
    replace_missing_values_with_constant(data, -100)
    drop_features(data, ['id', 'timestamp'])
    #data = principal_components_analysis(data, 80)
    data = attribute_subset_selection_with_trees(data)
    data = z_score_normalization(data)
    #data = min_max_scaler(data)



if __name__ == "__main__":
    data = open_file("../resources/train.csv")
    #first_iteration(data)
    second_iteration(data)
    write_file(data, "../resources/output.csv")
