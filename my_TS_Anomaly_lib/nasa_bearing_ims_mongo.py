
import os

import re

from pymongo import MongoClient, DESCENDING
import pandas as pd
from datetime import datetime

from joblib import parallel_backend, Parallel, delayed
from tqdm.notebook import tqdm as notebook_tqdm

from .utils import millify


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
    measurements_file
    , test_id
    , dump_size = 5_000

    , db_write_username = 'myAdmin'
    , password = 'my_admin_password'
) -> None :
    """
    Importing the content of 'measurements_file'
    into the Mongo DB database.
    Each batch-import contains at most 'dump_size' records.

    Parameters :
        - measurements_file (tuple(str, str)) :
            (dirpath, filename) of the sensors measurements file
            the content of which must be transferred.
        - dump_size (int) :
            max. number of documents to be written
            to the Mongo DB database at a time.
        - db_write_username (str) :
            name of the Mongo DB user used to operate the data dump
            (write perission on database 'nasa_ims_database' required)
        - password (str) :
            password of the Mongo DB user used to operate the data dump

    Results :
        - N.A.
    """

    with MongoClient(
        'mongodb://%s:%s@127.0.0.1' % (db_write_username, password)
    ) as client :

        db = client.nasa_ims_database
        coll = db.measurements

        (dirpath, filename) = measurements_file
        file_path = os.path.join(dirpath, filename)

        #####################################################
        # read the content of the sensors measurements file #
        #####################################################
        with open(file_path, 'rt') as measurements_file :
            #print('process connected ' + file_path)
            measurements = [line.rstrip('\n').split('\t')
                            for line in measurements_file]
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
                            , columns = ['sensor_' + str(j+1)
                                         for j in range(len(measurements_slice[0]))]
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

    Results :
        - N.A.
    """

    if not database_name in mongo_client.list_database_names() :
        print("Transferring data to Mongo DB :")
        db = mongo_client.nasa_ims_database
        coll = db.measurements

        silent = coll.create_index(
            [('test_id', DESCENDING)], name = 'test_id_idx', background = True
        )

        ########################################################
        # make an inventory of the files to be imported        #
        ########################################################
        measurements_files = []
        for (dirpath, dirnames, filenames) in os.walk(root_dir) :
            for filename in filenames : #[1:10] :
                if measurements_filename_pattern.match(filename) :
                    measurements_files.append((dirpath, filename))
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
                        measurements_files[idx]
                        , test_id = \
                            subfolders.index(measurements_files[idx][0]) + 1
                    )
                    for idx in range(len(measurements_files))
                )
        ########################################################

        print("Imported " +
              millify(
                  mongo_client.nasa_ims_database.measurements \
                  .estimated_document_count()) + " 'timestep' documents"
             )
    else :
        print("EXITED. A database named '"+ database_name +
              "' already exists on the Mongo DB server.")

    return


#/////////////////////////////////////////////////////////////////////////////////////




















































































































