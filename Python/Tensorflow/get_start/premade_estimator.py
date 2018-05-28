import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data(label_name='Species'):
    #导入数据
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

def describe_data(tran_x):
    my_feature_columns=[]
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns

def train_input_fn(features,labels,batch_size):
    dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    dataset=dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def input_fn(features,labels=None,batch_size=None):
    if labels is None:
        inputs=features
    else:
        inputs=(features,labels)
    
    dataset=tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None
    dataset=dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


    







(train_x,train_y),(test_x,test_y)=load_data()

my_feature_columns=describe_data(train_x)

classifier=tf.estimator.Estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=[10,10],n_calsses=3)

classifier.train(input_fn=train_input_fn(train_x,train_y),steps=args.train_steps)
