import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data(label_name='Species'):

    train_path = "D:\project\Python\Tensorflow\get_start\iris_training.csv"
    test_path="D:\project\Python\Tensorflow\get_start\iris_test.csv"
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

def Build_model():
    (train_x,train_y),(test_x,test_y)=pes.load_data()

    #Create feature columns for all features.
    my_feature_columns=[]
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    '''
    上述语句等价于
    my_feature_columns = [
     tf.feature_column.numeric_column(key='SepalLength'),
     tf.feature_column.numeric_column(key='SepalWidth'),
        tf.feature_column.numeric_column(key='PetalLength'),
        tf.feature_column.numeric_column(key='PetalWidth')
    ]
    '''
    '''
    Use the hidden_units parameter to define the number of neurons in each hidden layer of the neural network. Assign this parameter a list.
    The n_classes parameter specifies the number of possible values that the neural network can predict. Since the Iris problem classifies 3 Iris species, we set n_classes to 3.
    '''
    classifier=tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10,10],
        n_classes=3
        )

def train_input_fn(model,features, labels, batch_size):
    model.train()


#(train_features,train_label),(test_features,test_label)=load_data()
print("ok")