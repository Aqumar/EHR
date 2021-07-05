import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
# As the cardinality of the categorical feature variable ncd_code are too high. Mapping the ndc_code to Non-proprietary Name can help reduce the cardinality.
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_new = pd.merge(df, ndc_code_df[['NDC_Code', 'Non-proprietary Name']],
                      how="left",
                      left_on='ndc_code',
                      right_on='NDC_Code')
    df_new.rename(columns={"Non-proprietary Name": "generic_drug_name"}, inplace=True)
    
    return df_new

#Question 4
# The code is to select the first encounter data point for every patient from the set of unique patients. 
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    reduce_dim_df = df
    unique_patients = list(set(reduce_dim_df['patient_nbr']))
    unique_encounters = sorted(list(set(reduce_dim_df['encounter_id'])))
    sort_encounter_df = reduce_dim_df.sort_values(by = 'encounter_id').reset_index(drop=True)
    each_patient_first_encounter = []
    for patient_id in unique_patients:
    	each_patient_first_encounter.append(sort_encounter_df[sort_encounter_df['patient_nbr'] == patient_id].iloc[0])
    first_encounter_df = pd.concat(each_patient_first_encounter, axis=1)
    first_encounter_df = first_encounter_df.T
    first_encounter_df = first_encounter_df.reset_index(drop=True)

    return first_encounter_df


#Question 6
#Splitting the data into train, validation and test set such that there are no data leekage.
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    train_percentage = 0.6
    valid_percentage = 0.2
    test_percentage = 0.2

    unique_patient_ids = list(set(processed_df['patient_nbr']))
    random.shuffle(unique_patient_ids)

    train = unique_patient_ids[:round(len(unique_patient_ids)*train_percentage)]
    valid = unique_patient_ids[round(len(unique_patient_ids)*train_percentage):round(len(unique_patient_ids)*(train_percentage + valid_percentage))]
    test = unique_patient_ids[round(len(unique_patient_ids)*(train_percentage + valid_percentage)):]

    d_train = processed_df[processed_df['patient_nbr'].isin(train)]
    d_val = processed_df[processed_df['patient_nbr'].isin(valid)]
    d_test = processed_df[processed_df['patient_nbr'].isin(test)]
    
    return train, validation, test

#Question 7
#The function returns the list of one hot encoded categories using the tensorflow feature column API.
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
	
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
  
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
	tf_category = tf.feature_column.categorical_column_with_vocabulary_file(
                c, vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_category)
        output_tf_list.append(tf_categorical_feature_column)

    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std


#The function help to return the normalized numerical feature vector using the tensorflow feature column API.
def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    return tf.feature_column.numeric_column(key=col, default_value = 0, normalizer_fn=normalizer, dtype=tf.float64)

    #return tf_numeric_feature

#Question 9
#Returns the mean and standard deviation of the diabetes_yhat which is the tensorflow probability prediction from the model.
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
#Returns the mean prediction output from the model to the binary label based on threshold value 0.5
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = prob_output_df['pred_mean'].apply(lambda x: 1 if x>=5 else 0).to_numpy()
    return student_binary_prediction
