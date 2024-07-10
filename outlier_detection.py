# PyOD Script for Outlier and Anomaly Detection
# TODO: add novelty detection
# TODO: add test cases in a test file or folder
# TODO: add docs
# TODO: add setup
# TODO: packaging
# TODO: pip requirements
# TODO: add default model selections
# TODO: add logging, observability?
# TODO: upgrade python version after testing, speed comparison. Compiled to wasm, c, executable?
# TODO: add optimized model selection
# TODO: test Mojo and other Python runtimes and compilers
# TODO: alternate options for columns with missing data

# TODO: add optional visualization


import datetime
# import sys

import hydra
from omegaconf import DictConfig
import pandas as pd

# import polars as pl
# TODO: add more model types from PyOD
from pyod.models.suod import SUOD, LOF, IForest, COPOD
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="conf", config_name="config")
def start_outlier_detection(config: DictConfig):
    dataframe = load_dataframe(
        config.input_type,
        config.input_path,
        config.engine,
    )
    dataframe_numeric_only = filter_numeric_columns(dataframe)
    dataframe_missing_data_columns_removed = remove_missing_data_columns(dataframe_numeric_only)
    dataframe_with_outliers = outlier_detection(dataframe_missing_data_columns_removed)
    export_dataframe(config.engine, dataframe_with_outliers, config.output_type, config.output_path)


def filter_numeric_columns(dataframe):
    dataframe_numeric_only = dataframe.select_dtypes(include="number")
    return dataframe_numeric_only


def remove_missing_data_columns(dataframe):
    dataframe_missing_data_columns_removed = dataframe.dropna(axis=1)
    return dataframe_missing_data_columns_removed


def get_date():
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    return date

def get_time():
    time = datetime.datetime.now().strftime("%H:%M:%S")
    return time


# TODO: add sql and other file formats later
# TODO: add option for polars vs pandas
# TODO: add option for multiple import? Or have separate function for multiple import?
def load_dataframe(
    input_type, input_path, polars_or_pandas, connection_string=None, query=None
):
    # TODO: switch to case from if else
    if input_type == "csv":
        dataframe = pd.read_csv(input_path)
    elif input_type == "excel":
        dataframe = pd.read_excel(input_path)
    # elif input_type == "fwf":
    #     dataframe = pd.read_fwf(input_path)
    # elif input_type == "json":
    #     dataframe = pd.read_json(input_path)
    # elif input_type == "sql_query":
    #     dataframe = pd.read_sql_query(input_path)
    # elif input_type == "sql_table":
    #     dataframe = pd.read_sql_table(input_path)
    # elif input_type == "sql":
    #     dataframe = pd.read_sql(input_path)
    return dataframe


# TODO: add more advanced file naming, for example: data asset name + input data set name + date + time + model name + output type
# TODO: add sql and other file formats later
# TODO: add option for polars vs pandas
# TODO: add option for multiple export? Or have separate function for multiple export?
def export_dataframe(
    polars_or_pandas,
    dataframe,
    export_type,
    export_path,
    connection_string=None,
    query=None,
):
    # TODO: add option for metadata, data asset name + input data set name + date + time + model name + output type
    # TODO: figure out good option for different output types
    # export_filepath = get_date() + "_" + get_time() + "_" + export_path

    if export_type == "csv":
        dataframe.to_csv(export_path)
    elif export_type == "excel":
        dataframe.to_excel(export_path)
    # elif export_type == "fwf":
    #     dataframe.to_fwf(export_path)
    # elif export_type == "json":
    #     dataframe.to_json(export_path)
    # elif export_type == "sql":
    #     dataframe.to_sql(export_path)
    # elif export_type == "sql_query":
    #     dataframe.to_sql_query(export_path)
    # elif export_type == "sql_table":
    #     dataframe.to_sql_table(export_path)


def combine_train_test_outliers(train, train_pred, train_scores, test, test_pred, test_scores):
    train["pred"] = train_pred
    train["scores"] = train_scores
    test["pred"] = test_pred
    test["scores"] = test_scores
    recombined_dataframe = pd.concat([train, test], axis=0)
 
    print("recombined_dataframe shape: ", recombined_dataframe.shape)
    return recombined_dataframe


# def combine_classifiers(dataframe):
#     print("Combining classifiers...")
    # pyod.models.combination module
    # pyod.models.combination.aom(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)[source]
    # pyod.models.combination.average(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)[source]
    # pyod.models.combination.maximization(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)[source]
    # pyod.models.combination.median(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)[source]
    # pyod.models.combination.majority_vote(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)[source]
    # pyod.models.combination.moa(scores, n_buckets=5, method='static', bootstrap_estimators=False, random_state=None)[source]


def outlier_detection(dataframe, train_test_split_ratio=0.3, model_type=["SUOD"]):
    train, test = train_test_split(dataframe, test_size=train_test_split_ratio)

    # initialized a group of outlier detectors for acceleration
    detector_list = [
        LOF(n_neighbors=15),
        LOF(n_neighbors=20),
        LOF(n_neighbors=25),
        LOF(n_neighbors=35),
        COPOD(),
        IForest(n_estimators=100),
        IForest(n_estimators=200),
    ]

    # decide the number of parallel process, and the combination method
    clf = SUOD(
        base_estimators=detector_list, n_jobs=2, combination="average", verbose=False
    )

    # or to use the default detectors
    # clf = SUOD(n_jobs=2, combination='average',
    #            verbose=False)
    clf.fit(train)

    # get the prediction labels and outlier scores of the training data
    train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    test_pred = clf.predict(test)  # outlier labels (0 or 1)
    test_scores = clf.decision_function(test)  # outlier scores

    dataframe_plus_outliers = combine_train_test_outliers(train, train_pred, train_scores, test, test_pred, test_scores)
    return dataframe_plus_outliers


def main():
    start_outlier_detection()


if __name__ == "__main__":
    main()


# def main():
#     try:
#         start_outlier_detection()
#     except ValueError as ve:
#         return ve

# if __name__ == "__main__":
#     sys.exit(main())
