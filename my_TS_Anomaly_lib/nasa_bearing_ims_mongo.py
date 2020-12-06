
import os, re
import pandas as pd

from pymongo import MongoClient, DESCENDING, ASCENDING
from datetime import datetime, timedelta
from pprint import pprint

from joblib import parallel_backend, Parallel, delayed
from tqdm.notebook import tqdm as notebook_tqdm

from .utils import millify, print_timedelta


#/////////////////////////////////////////////////////////////////////////////////////


class ProgressParallel(Parallel):
    """
    Extension to the 'joblib.Parallel' class,
    allowing for a tqdm progressbar followup.
    """
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with notebook_tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


#/////////////////////////////////////////////////////////////////////////////////////


def import_file_content(
    database_name
    , measurements_file
    , test_id
    , dump_size = 5_000

    , db_write_username = 'myAdmin'
    , password = 'my_admin_password'

    , dev_env = False
) -> None :
    """
    Importing the content of 'measurements_file'
    into the Mongo DB database.
    Each batch-import contains at most 'dump_size' records.

    Parameters :
        - database_name (str) :
            the name of the database to be created
        - measurements_file (tuple(str, str)) :
            (dirpath, filename) of the sensors measurements file
            the content of which must be transferred.
        - dump_size (int) :
            max. number of documents to be written
            to the Mongo DB database at a time.
        - db_write_username (str) :
            name of the Mongo DB user used to operate the data dump
            (write perission on database 'database_name' required)
        - password (str) :
            password of the Mongo DB user used to operate the data dump
        - dev_env (bool) :
            whether or not the database collection to be created
            shall consider only part of the source data
            (i.e. if it is for use in a 'development environment' context),
            in which case only one every 1000 measurements is considered.

    Results :
        - N.A.
    """

    with MongoClient(
        'mongodb://%s:%s@127.0.0.1' % (db_write_username, password)
    ) as client :

        db = client[database_name]
        coll = db.measurements

        (dirpath, filename) = measurements_file
        file_path = os.path.join(dirpath, filename)

        #####################################################
        # read the content of the sensors measurements file #
        #####################################################
        if not dev_env :
            increment = 1
        else :
            increment = 1_000
        with open(file_path, 'rt') as measurements_file_handle :
            measurements = [
                [int(lineno / increment)]+
                [
                    float(i) for i in line.rstrip('\n').split('\t')
                ]
                for lineno, line in enumerate(measurements_file_handle)
                if (not dev_env) | (lineno % increment == 0)
            ]
        #####################################################


        #####################################################
        # loop over batches of max. size 'dump_size'        #
        # and import to Mongo DB                            #
        #####################################################
        batches_count = -(-len(measurements) // dump_size)
        for batch_nbr in range(batches_count) :
            #print("-" + str(len(measurements[ batch_nbr*dump_size : (batch_nbr+1)*dump_size ])))
            measurements_slice = \
                measurements[ batch_nbr*dump_size : (batch_nbr+1)*dump_size ]
            coll.insert_many(
                pd.concat(
                    [
                        pd.DataFrame({'test_id':
                                          [
                                              test_id
                                          ] * len(measurements_slice)
                                      , 'timestamp':
                                          [
                                              datetime.strptime(filename, '%Y.%m.%d.%H.%M.%S')
                                          ] * len(measurements_slice)})
                        , pd.DataFrame(
                            measurements_slice
                            , columns = ['step']+['sensor_' + str(j+1)
                                         for j in range(len(measurements_slice[0])-1)]
                        )
                    ]
                    , axis = 1
                ).to_dict('records')
            )
        #####################################################

    return


#/////////////////////////////////////////////////////////////////////////////////////


measurements_filename_pattern = \
    re.compile('^200[3-4]\\.\d{2}\\.\d{2}\\.\d{2}\\.\d{2}\\.\d{2}$')


def measurements_to_mongo(
    mongo_client
    , database_name
    , root_dir

    , par_backend: str
    , n_jobs: int

    , dev_env = False
) -> None :
    """
    Transfering the content of the sensors measurements files
    from a local directory structure into a new Mongo DB database
    using parallel processing (with progress bar).

    Parameters :
        - mongo_client (pymongo.MongoClient) :
            the Mongo DB client (connected to the target server)
        - database_name (str) :
            the name of the database to be created
        - root_dir (str) :
            the path to the parent directory
            where the sensors measurements files are stored
        - par_backend (str) :
            the parallel backend to be used
            (by joblib.Parallel). Can be either
            'loky', 'threading', or 'multiprocessing'.
        - n_jobs (int) :
            the maximum number of concurrently running jobs
        - dev_env (bool) :
            whether or not the database collection to be created
            shall consider only part of the source data
            (i.e. if it is for use in a 'development environment' context)
            in which case the first 3/4 of measurement files are excluded.

    Results :
        - N.A.
    """

    if not database_name in mongo_client.list_database_names() :
        print("Transferring data to Mongo DB :")
        db = mongo_client[database_name]
        coll = db.measurements

        silent = coll.create_index(
            [('test_id', DESCENDING)], name = 'test_id_idx', background = True
        )
        silent = coll.create_index(
            [('timestamp', DESCENDING)], name = 'timestamp_idx', background = True
        )

        ########################################################
        # make an inventory of the files to be imported        #
        ########################################################
        measurements_files = []
        for (dirpath, dirnames, filenames) in os.walk(root_dir) :
            for filename in filenames : #[:10] :
                if measurements_filename_pattern.match(filename) :
                    measurements_files.append((dirpath, filename))
        if dev_env :
            measurements_files = measurements_files[
                int((3/4)*len(measurements_files))::
            ]
        ########################################################
        subfolders = \
            sorted(
                list(set([dirpath for (dirpath, _) in measurements_files]))
                , reverse=False)
        #print(subfolders)

        ########################################################
        # import the content of the sensors measurements files #
        # to the Mongo DB database                             #
        ########################################################
        verbose = 1
        total = len(measurements_files)
        if total > 0 :
            with parallel_backend(par_backend) :
                result_list = ProgressParallel(n_jobs = n_jobs
                                               , use_tqdm = (verbose > 0)
                                               , total = total)(
                    delayed(import_file_content)(
                        database_name
                        , measurements_files[idx]
                        , test_id = \
                            subfolders.index(measurements_files[idx][0]) + 1
                        , dev_env = dev_env
                    )
                    for idx in range(len(measurements_files))
                )
        ########################################################

        print("Imported " +
              millify(
                  db.measurements \
                  .estimated_document_count()) + " 'timestep' documents."
             )
    else :
        print("EXITED. A database named '"+ database_name +
              "' already exists on the Mongo DB server.")

    return


#/////////////////////////////////////////////////////////////////////////////////////


def get_test_first_timestamps(
    mongo_client: MongoClient
    , database_name: str = 'nasa_ims_database' # 'nasa_ims_database_dev' # 
    , test_id: int = 2
    , timestamps_count: int = 6
    , sensor_name: str = 'sensor_1'
) :
    """
    Retrieve the data points corresponding to a given sensor for a given test.
    Only collects data points for some earliest measurements.

    Parameters :
        - mongo_client (pymongo.MongoClient) :
            the Mongo DB client (connected to the target server)
        - database_name (str) :
            the name of the database to be created
        - test_id (int) :
            which one of the tests shall be considered
        - timestamps_count (int) :
            how many of the earliest timestamps
            shall be considered
        - sensor_name (str) :
            which one of the sensors shall be considered
    Results :
        - (pd.DataFrame) :
            retrieved measurement records
            The recordset is enriched with a 'long_datetime' column
            gathering the information from the 'timestamp'
            and the 'step' fields.
            REMINDER :
                each measurement consists of
                several data points (several steps).
    """

    start_time = datetime.now()

    #######################################
    # RETRIEVE THE FIRST TIMESTAMP VALUES #
    #######################################
    cursor = mongo_client[database_name].measurements.aggregate([
        {'$match': { 'test_id': test_id }}
        , {'$group': { '_id': "$timestamp"}}
        , {'$sort': {'_id': ASCENDING}}
        , {'$limit': timestamps_count}
    ])
    earliest_timestamps = list(cursor)
    #pprint(earliest_timestamps)
    #######################################

    ##########################################
    # RETRIEVE THE CORRESPONDING MEASURMENTS #
    ##########################################
    cursor = mongo_client[database_name].measurements.aggregate([
        {'$match': {
            'test_id': test_id
            , "timestamp": {
               "$gte": earliest_timestamps[0].get('_id')
               , "$lte": earliest_timestamps[-1].get('_id')
            }
        }}
        , {'$sort': {'step': ASCENDING}}
        , { "$project": {
            "_id": 0
            , "timestamp": 1
            , "step": 1
            , sensor_name: 1
        } }
    ])
    data = pd.DataFrame(list(cursor))
    ##########################################

    # format according to the number of different steps
    # of each "1-second-long" measurement =>
    second_slices_count = int(data.shape[0]/timestamps_count) - 1
    def datetime_slice_function(df_row) :
        return df_row['timestamp'] + \
               timedelta(seconds = df_row['step']/second_slices_count)
    data['long_datetime'] = data[['timestamp', 'step']].apply(datetime_slice_function, axis=1)
    del data['step']


    print_timedelta(start_time)
    return data


#/////////////////////////////////////////////////////////////////////////////////////

















































































































