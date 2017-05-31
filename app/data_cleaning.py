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

def replace_outliers_with_null(data, feature, outlier_value):
    outliers = data.loc[data[feature] >= outlier_value, feature].index
    data.ix[outliers, feature] = np.NaN
    return data


def check_for_inconsistencies(data):
    life_sq_greater_full_sq = data[data.life_sq > data.full_sq].index
    data.ix[life_sq_greater_full_sq, "life_sq"] = np.NaN
    life_sq_less_5 = data[data.life_sq < 5].index
    data.ix[life_sq_less_5, "life_sq"] = np.NaN
    full_sq_less_5 = data[data.full_sq < 5].index
    data.ix[full_sq_less_5, "life_sq"] = np.NaN
    kitch_sq_greater_life_sq = data[data.kitch_sq >= data.life_sq].index
    data.ix[kitch_sq_greater_life_sq, "kitch_sq"] = np.NaN
    kitch_sq_small_val = data[(data.kitch_sq == 0).values + (data.kitch_sq == 1).values].index
    data.ix[kitch_sq_small_val, "kitch_sq"] = np.NaN
    build_year_small_val = data[data.build_year < 1500].index
    data.ix[build_year_small_val, "build_year"] = np.NaN
    build_year_big_val = data[data.build_year > 2020].index
    data.ix[build_year_big_val, "build_year"] = np.NaN
    num_room_0 = data[data.num_room == 0].index
    data.ix[num_room_0, "num_room"] = np.NaN
    floor_0 = data[data.floor == 0].index
    data.ix[floor_0, "floor"] = np.NaN
    max_floor_0 = data[data.max_floor == 0].index
    data.ix[max_floor_0, "max_floor"] = np.NaN
    floor_greater_max_floor = data[data.floor > data.max_floor].index
    data.ix[floor_greater_max_floor, "max_floor"] = np.NaN

    return data

def feature_engineering(data):
    columns_length = len(data.columns) - 2

    # ratio of living area to full area #
    data.insert(columns_length, "ratio_life_sq_full_sq", data["life_sq"] / np.maximum(data["full_sq"].astype("float"),1))
    data["ratio_life_sq_full_sq"].ix[data["ratio_life_sq_full_sq"]<0] = 0
    data["ratio_life_sq_full_sq"].ix[data["ratio_life_sq_full_sq"]>1] = 1

    # ratio of kitchen area to living area #
    data.insert(columns_length, "ratio_kitch_sq_life_sq", data["kitch_sq"] / np.maximum(data["life_sq"].astype("float"),1))
    data["ratio_kitch_sq_life_sq"].ix[data["ratio_kitch_sq_life_sq"]<0] = 0
    data["ratio_kitch_sq_life_sq"].ix[data["ratio_kitch_sq_life_sq"]>1] = 1

    # ratio of kitchen area to full area #
    data.insert(columns_length, "ratio_kitch_sq_full_sq", data["kitch_sq"] / np.maximum(data["full_sq"].astype("float"),1))
    data["ratio_kitch_sq_full_sq"].ix[data["ratio_kitch_sq_full_sq"]<0] = 0
    data["ratio_kitch_sq_full_sq"].ix[data["ratio_kitch_sq_full_sq"]>1] = 1

    # floor of the house to the total number of floors in the house #
    data.insert(columns_length, "ratio_floor_max_floor", data["floor"] / data["max_floor"].astype("float"))

    # num of floor from top #
    data.insert(columns_length, "floor_from_top", data["max_floor"] - data["floor"])

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
    extra_tree = ExtraTreesClassifier(n_estimators=300, max_features=20, max_depth=10)

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

def second_iteration(train, test):
    remove_outliers(train, 'full_sq', 5000)
    remove_outliers(train, 'life_sq', 7000)
    remove_outliers(train, 'floor', 70)
    remove_outliers(train, 'max_floor', 80)
    remove_outliers(train, 'num_room', 15)
    remove_outliers(train, 'kitch_sq', 1750)
    remove_outliers(train, 'state', 30)
    remove_outliers(train, 'children_preschool', 17500)
    remove_outliers(train, 'preschool_education_centers_raion', 12)
    remove_outliers(train, 'children_school', 17500)
    remove_outliers(train, 'healthcare_centers_raion', 5)
    remove_outliers(train, 'additional_education_raion', 14)
    remove_outliers(train, 'culture_objects_top_25_raion', 8)
    remove_outliers(train, 'office_raion', 120)
    remove_outliers(train, 'young_all', 40000)
    remove_outliers(train, 'young_male', 20000)
    remove_outliers(train, 'young_female', 17500)
    remove_outliers(train, '0_6_all', 17500)
    remove_outliers(train, '0_6_male', 8000)
    remove_outliers(train, '0_6_female', 8000)
    remove_outliers(train, '7_14_all', 17500)
    remove_outliers(train, '0_17_male', 20000)
    remove_outliers(train, '0_17_female', 20000)
    remove_outliers(train, '0_13_all', 35000)
    remove_outliers(train, '0_13_male', 12500)
    remove_outliers(train, '0_13_female', 15000)
    remove_outliers(train, 'raion_build_count_with_material_info', 1500)
    remove_outliers(train, 'build_count_monolith', 100)
    remove_outliers(train, 'build_count_panel', 400)
    remove_outliers(train, 'build_count_foam', 10)
    remove_outliers(train, 'build_count_slag', 80)
    remove_outliers(train, 'raion_build_count_with_builddate_info', 1500)
    remove_outliers(train, 'build_count_before_1920', 350)
    remove_outliers(train, 'build_count_1921-1945', 350)
    remove_outliers(train, 'build_count_1946-1970', 800)
    remove_outliers(train, 'build_count_1971-1995', 200)
    remove_outliers(train, 'build_count_after_1995', 700)
    remove_outliers(train, 'school_km', 40)
    remove_outliers(train, 'park_km', 40)
    remove_outliers(train, 'industrial_km', 12)
    remove_outliers(train, 'bus_terminal_avto_km', 70)
    remove_outliers(train, 'stadium_km', 80)
    remove_outliers(train, 'university_km', 80)
    remove_outliers(train, 'additional_education_km', 20)
    remove_outliers(train, 'big_church_km', 40)
    remove_outliers(train, 'church_synagogue_km', 14)
    remove_outliers(train, 'mosque_km', 40)
    remove_outliers(train, 'trc_sqm_500', 1400000)
    remove_outliers(train, 'cafe_sum_500_min_price_avg', 3500)
    remove_outliers(train, 'cafe_sum_500_max_price_avg', 5000)
    remove_outliers(train, 'cafe_avg_price_500', 4000)
    remove_outliers(train, 'mosque_count_500', 1.0)
    remove_outliers(train, 'trc_sqm_1000', 1400000)
    remove_outliers(train, 'mosque_count_1000', 1.0)
    remove_outliers(train, 'mosque_count_1500', 1.0)
    remove_outliers(train, 'trc_sqm_2000', 2000000)
    remove_outliers(train, 'mosque_count_2000', 1.0)
    remove_outliers(train, 'material', 4)
    remove_outliers(train, 'build_year', 2020)
    bad_index = train.loc[train['build_year'] < 1500].index
    train.drop(bad_index, inplace=True)
    remove_outliers(train, 'green_zone_part', 0.7)
    remove_outliers(train, 'indust_part', 0.45)
    remove_outliers(train, 'preschool_quota', 7000)
    remove_outliers(train, 'school_quota', 16000)
    remove_outliers(train, 'school_education_centers_top_20_raion', 0.75)
    remove_outliers(train, 'hospital_beds_raion', 3500)
    remove_outliers(train, 'university_top_20_raion', 3)
    nominal_to_numeric(train)
    nominal_to_numeric(test)
    replace_missing_values_with_constant(train, -100)
    replace_missing_values_with_constant(test, -100)
    drop_features(train, ['id', 'timestamp'])
    #data = principal_components_analysis(data, 80)
    train = attribute_subset_selection_with_trees(train)
    test = attribute_subset_selection_with_trees(test)
    train = z_score_normalization(train)
    test = z_score_normalization(test)
    #data = min_max_scaler(data)

def third_iteration(train, test):
    replace_outliers_with_null(train, 'full_sq', 5000)
    replace_outliers_with_null(train, 'life_sq', 7000)
    replace_outliers_with_null(train, 'floor', 70)
    replace_outliers_with_null(train, 'max_floor', 80)
    replace_outliers_with_null(train, 'num_room', 15)
    replace_outliers_with_null(train, 'kitch_sq', 1750)
    replace_outliers_with_null(train, 'state', 30)
    replace_outliers_with_null(train, 'children_preschool', 17500)
    replace_outliers_with_null(train, 'preschool_education_centers_raion', 12)
    replace_outliers_with_null(train, 'children_school', 17500)
    replace_outliers_with_null(train, 'healthcare_centers_raion', 5)
    replace_outliers_with_null(train, 'additional_education_raion', 14)
    replace_outliers_with_null(train, 'culture_objects_top_25_raion', 8)
    replace_outliers_with_null(train, 'office_raion', 120)
    replace_outliers_with_null(train, 'young_all', 40000)
    replace_outliers_with_null(train, 'young_male', 20000)
    replace_outliers_with_null(train, 'young_female', 17500)
    replace_outliers_with_null(train, '0_6_all', 17500)
    replace_outliers_with_null(train, '0_6_male', 8000)
    replace_outliers_with_null(train, '0_6_female', 8000)
    replace_outliers_with_null(train, '7_14_all', 17500)
    replace_outliers_with_null(train, '0_17_male', 20000)
    replace_outliers_with_null(train, '0_17_female', 20000)
    replace_outliers_with_null(train, '0_13_all', 35000)
    replace_outliers_with_null(train, '0_13_male', 12500)
    replace_outliers_with_null(train, '0_13_female', 15000)
    replace_outliers_with_null(train, 'raion_build_count_with_material_info', 1500)
    replace_outliers_with_null(train, 'build_count_monolith', 100)
    replace_outliers_with_null(train, 'build_count_panel', 400)
    replace_outliers_with_null(train, 'build_count_foam', 10)
    replace_outliers_with_null(train, 'build_count_slag', 80)
    replace_outliers_with_null(train, 'raion_build_count_with_builddate_info', 1500)
    replace_outliers_with_null(train, 'build_count_before_1920', 350)
    replace_outliers_with_null(train, 'build_count_1921-1945', 350)
    replace_outliers_with_null(train, 'build_count_1946-1970', 800)
    replace_outliers_with_null(train, 'build_count_1971-1995', 200)
    replace_outliers_with_null(train, 'build_count_after_1995', 700)
    replace_outliers_with_null(train, 'school_km', 40)
    replace_outliers_with_null(train, 'park_km', 40)
    replace_outliers_with_null(train, 'industrial_km', 12)
    replace_outliers_with_null(train, 'bus_terminal_avto_km', 70)
    replace_outliers_with_null(train, 'stadium_km', 80)
    replace_outliers_with_null(train, 'university_km', 80)
    replace_outliers_with_null(train, 'additional_education_km', 20)
    replace_outliers_with_null(train, 'big_church_km', 40)
    replace_outliers_with_null(train, 'church_synagogue_km', 14)
    replace_outliers_with_null(train, 'mosque_km', 40)
    replace_outliers_with_null(train, 'trc_sqm_500', 1400000)
    replace_outliers_with_null(train, 'cafe_sum_500_min_price_avg', 3500)
    replace_outliers_with_null(train, 'cafe_sum_500_max_price_avg', 5000)
    replace_outliers_with_null(train, 'cafe_avg_price_500', 4000)
    replace_outliers_with_null(train, 'mosque_count_500', 1.0)
    replace_outliers_with_null(train, 'trc_sqm_1000', 1400000)
    replace_outliers_with_null(train, 'mosque_count_1000', 1.0)
    replace_outliers_with_null(train, 'mosque_count_1500', 1.0)
    replace_outliers_with_null(train, 'trc_sqm_2000', 2000000)
    replace_outliers_with_null(train, 'mosque_count_2000', 1.0)
    replace_outliers_with_null(train, 'build_year', 2020)
    replace_outliers_with_null(train, 'green_zone_part', 0.7)
    replace_outliers_with_null(train, 'indust_part', 0.45)
    replace_outliers_with_null(train, 'preschool_quota', 7000)
    replace_outliers_with_null(train, 'school_quota', 16000)
    replace_outliers_with_null(train, 'school_education_centers_top_20_raion', 0.75)
    replace_outliers_with_null(train, 'hospital_beds_raion', 3500)
    replace_outliers_with_null(train, 'university_top_20_raion', 3)
    nominal_to_numeric(train)
    check_for_inconsistencies(train)
    feature_engineering(train)
    replace_missing_values_with_constant(train, -100)
    train = attribute_subset_selection_with_trees(train)
    train = z_score_normalization(train)
    replace_outliers_with_null(test, 'full_sq', 5000)
    replace_outliers_with_null(test, 'life_sq', 7000)
    replace_outliers_with_null(test, 'floor', 70)
    replace_outliers_with_null(test, 'max_floor', 80)
    replace_outliers_with_null(test, 'num_room', 15)
    replace_outliers_with_null(test, 'kitch_sq', 1750)
    replace_outliers_with_null(test, 'state', 30)
    replace_outliers_with_null(test, 'children_preschool', 17500)
    replace_outliers_with_null(test, 'preschool_education_centers_raion', 12)
    replace_outliers_with_null(test, 'children_school', 17500)
    replace_outliers_with_null(test, 'healthcare_centers_raion', 5)
    replace_outliers_with_null(test, 'additional_education_raion', 14)
    replace_outliers_with_null(test, 'culture_objects_top_25_raion', 8)
    replace_outliers_with_null(test, 'office_raion', 120)
    replace_outliers_with_null(test, 'young_all', 40000)
    replace_outliers_with_null(test, 'young_male', 20000)
    replace_outliers_with_null(test, 'young_female', 17500)
    replace_outliers_with_null(test, '0_6_all', 17500)
    replace_outliers_with_null(test, '0_6_male', 8000)
    replace_outliers_with_null(test, '0_6_female', 8000)
    replace_outliers_with_null(test, '7_14_all', 17500)
    replace_outliers_with_null(test, '0_17_male', 20000)
    replace_outliers_with_null(test, '0_17_female', 20000)
    replace_outliers_with_null(test, '0_13_all', 35000)
    replace_outliers_with_null(test, '0_13_male', 12500)
    replace_outliers_with_null(test, '0_13_female', 15000)
    replace_outliers_with_null(test, 'raion_build_count_with_material_info', 1500)
    replace_outliers_with_null(test, 'build_count_monolith', 100)
    replace_outliers_with_null(test, 'build_count_panel', 400)
    replace_outliers_with_null(test, 'build_count_foam', 10)
    replace_outliers_with_null(test, 'build_count_slag', 80)
    replace_outliers_with_null(test, 'raion_build_count_with_builddate_info', 1500)
    replace_outliers_with_null(test, 'build_count_before_1920', 350)
    replace_outliers_with_null(test, 'build_count_1921-1945', 350)
    replace_outliers_with_null(test, 'build_count_1946-1970', 800)
    replace_outliers_with_null(test, 'build_count_1971-1995', 200)
    replace_outliers_with_null(test, 'build_count_after_1995', 700)
    replace_outliers_with_null(test, 'school_km', 40)
    replace_outliers_with_null(test, 'park_km', 40)
    replace_outliers_with_null(test, 'industrial_km', 12)
    replace_outliers_with_null(test, 'bus_terminal_avto_km', 70)
    replace_outliers_with_null(test, 'stadium_km', 80)
    replace_outliers_with_null(test, 'university_km', 80)
    replace_outliers_with_null(test, 'additional_education_km', 20)
    replace_outliers_with_null(test, 'big_church_km', 40)
    replace_outliers_with_null(test, 'church_synagogue_km', 14)
    replace_outliers_with_null(test, 'mosque_km', 40)
    replace_outliers_with_null(test, 'trc_sqm_500', 1400000)
    replace_outliers_with_null(test, 'cafe_sum_500_min_price_avg', 3500)
    replace_outliers_with_null(test, 'cafe_sum_500_max_price_avg', 5000)
    replace_outliers_with_null(test, 'cafe_avg_price_500', 4000)
    replace_outliers_with_null(test, 'mosque_count_500', 1.0)
    replace_outliers_with_null(test, 'trc_sqm_1000', 1400000)
    replace_outliers_with_null(test, 'mosque_count_1000', 1.0)
    replace_outliers_with_null(test, 'mosque_count_1500', 1.0)
    replace_outliers_with_null(test, 'trc_sqm_2000', 2000000)
    replace_outliers_with_null(test, 'mosque_count_2000', 1.0)
    replace_outliers_with_null(test, 'build_year', 2020)
    replace_outliers_with_null(test, 'green_zone_part', 0.7)
    replace_outliers_with_null(test, 'indust_part', 0.45)
    replace_outliers_with_null(test, 'preschool_quota', 7000)
    replace_outliers_with_null(test, 'school_quota', 16000)
    replace_outliers_with_null(test, 'school_education_centers_top_20_raion', 0.75)
    replace_outliers_with_null(test, 'hospital_beds_raion', 3500)
    replace_outliers_with_null(test, 'university_top_20_raion', 3)
    nominal_to_numeric(test)
    check_for_inconsistencies(test)
    feature_engineering(test)
    replace_missing_values_with_constant(test, -100)
    test = attribute_subset_selection_with_trees(test)
    test = z_score_normalization(test)

if __name__ == "__main__":
    train = open_file("../resources/train.csv")
    test = open_file("../resources/test.csv")
    #first_iteration(data)
    #second_iteration(train, test)
    third_iteration(train, test)
    write_file(train, "../resources/train_output.csv")
    write_file(test, "../resources/test_output.csv")
