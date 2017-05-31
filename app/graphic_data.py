import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def find_outliers(data):
    numeric_columns = list(data.ix[1:2,data.dtypes!=object].columns)
    for i in numeric_columns:
        print(i)
        print(data[i].describe())

    for i in numeric_columns:
        plt.figure()
        plt.clf()
        sns.boxplot(data[i]);
        plt.title(i)
        plt.show()

    '''
    candidates for outliers
    full_sq - 5000
    life_sq - 7000
    floor - 70
    max_floor - 80
    num_room - 15
    kitch_sq - 1750
    state - 30
    children_preschool - 17500
    preschool_education_centers_raion - 12
    children_school - 17500
    healthcare_centers_raion - 5
    additional_education_raion - 14
    culture_objects_top_25_raion - 8
    office_raion - 120
    young_all - 40000
    young_male - 20000
    young_female - 17500
    0_6_all - 17500
    0_6_male - 8000
    0_6_female - 8000
    7_14_all - 17500
    7_14_male - 6000
    7_14_female - 8000
    0_17_all - 40000
    0_17_male - 20000
    0_17_female - 20000
    0_13_all - 35000
    0_13_male - 12500
    0_13_female - 15000
    raion_build_count_with_material_info - 1500
    build_count_monolith - 100
    build_count_panel - 400
    build_count_foam - 10
    build_count_slag - 80
    raion_build_count_with_builddate_info - 1500
    build_count_before_1920 - 350
    build_count_1921-1945 - 350
    build_count_1946-1970 - 800
    build_count_1971-1995 - 200
    build_count_after_1995 - 700
    school_km - 40
    park_km - 40
    industrial_km - 12
    bus_terminal_avto_km - 70
    stadium_km - 80
    university_km - 80
    additional_education_km - 20
    big_church_km - 40
    church_synagogue_km - 14
    mosque_km - 40
    trc_sqm_500 - 1400000
    cafe_sum_500_min_price_avg - 3500
    cafe_sum_500_max_price_avg - 5000
    cafe_avg_price_500 - 4000
    mosque_count_500 - 1.0
    trc_sqm_1000 - 1400000
    mosque_count_1000 - 1.0
    mosque_count_1500 - 1.0
    trc_sqm_2000 - 2000000
    mosque_count_2000 - 1.0
    '''

    '''
    candidates for outliers 2
    material - 4
    build_year - < 1500 y 2020 >
    green_zone_part - 0.7
    indust_part - 0.45
    preschool_quota - 7000
    school_quota - 16000
    school_education_centers_top_20_raion - 0.75
    hospital_beds_raion - 3500
    university_top_20_raion - 3
    '''

    '''
    candidates for outliers 3

    '''

def boxplot_graph(data, feature):
    ax = sns.boxplot(data[feature]);
    plt.show()

def find_num_of_missing_values(data):
    missing_df = data.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count']>0]
    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12,18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()

if __name__ == '__main__':
    data = open_file("../resources/train_output.csv")
    #find_outliers(data)
    #find_num_of_missing_values(data)
    boxplot_graph(data, 'full_sq')
