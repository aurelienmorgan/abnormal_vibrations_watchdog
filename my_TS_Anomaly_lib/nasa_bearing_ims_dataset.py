import os, sys

import tqdm

import requests, math, datetime

import py7zr
from rarfile import RarFile, BadRarFile


#/////////////////////////////////////////////////////////////////////////////////////


def download_progress(src_file_url, trgt_file_fullname, chunk_size=4096) :
    '''
    downloads a file from the Internet into a local file,
    with progress bar.

    parameters :
      - src_file_url (string) : url to the source file
      - file_fullname (string) : full path to the local file
      - chunk_size (int) : size of the chunk in bytes
    '''

    try:
        response = requests.get(src_file_url, stream=True)
        total_length = response.headers.get('content-length')
        total_length = int(total_length)
        steps = math.ceil(total_length/chunk_size)

        with tqdm.tqdm(
            total=steps
            , bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}'
        ) as pbar:
            with open(trgt_file_fullname,'wb') as f:
                for buffer in response.iter_content(
                    chunk_size=chunk_size
                ):
                    f.write( buffer )
                    pbar.update(1)
    except Exception as ex:
        raise


#/////////////////////////////////////////////////////////////////////////////////////


def download_nasa_ims_bearings_dataset(
    root_dir
) -> None :
    """
    Download the IMS bearings dataset from the NASA Repository
    and extract the compressed source files.

    Parameters :
        - root_dir (str) :
            local parent directory to be used
            to store the source files
    Results :
        - N.A.
    """

    grand_start_time = datetime.datetime.now()

    if not os.path.exists(root_dir) : os.makedirs(root_dir)

    url = "https://ti.arc.nasa.gov/c/3/"
    zipfullname = os.path.join( root_dir, 'IMS.7z')

    filenames = ['1st_test.rar', '2nd_test.rar', '3rd_test.rar'
                 , 'Readme Document for IMS Bearing Data.pdf']
    foldernames = {'1st_test.rar': '1st_test', '2nd_test.rar' : '2nd_test'
                   , '3rd_test.rar': '4th_test'}

    ##############################
    # download from remote host  #
    ##############################
    if sum([os.path.isfile(os.path.join( root_dir, filename))
            for filename in filenames]) != len(filenames) :
        # if any expected file is missing locally
        if not os.path.isfile(zipfullname) :
            print("Downloading archive :", file=sys.stderr, flush=True)
            download_progress(url , zipfullname)

        with py7zr.SevenZipFile(zipfullname, mode='r') as z:
            ERR_MESSAGE = \
                zipfullname + " seems corrupted as " + \
                "we can't locate expected entries"
            assert len([filename for filename in filenames
                        if filename in z.getnames()]) == len(filenames) \
                   , ERR_MESSAGE

            missing_filenames = [filename for filename in filenames
                                 if not os.path.isfile(os.path.join(root_dir, filename))]
            print("Extracting " + str(len(missing_filenames)) + " entries :\n" +
                  "\t" + str(missing_filenames), end='', file=sys.stderr, flush=True)
            start_time = datetime.datetime.now()
            z.extract(path = root_dir, targets=missing_filenames)
            timedelta = datetime.timedelta(seconds=(datetime.datetime.now()
                                                    - start_time).total_seconds())
            timedelta_str = \
                ':'.join(['{:02d}'.format(int(float(i)))
                          for i in str(timedelta).split(':')[:3]])
            print(" done [" + timedelta_str + "]."
                  , end='\n', file=sys.stderr, flush=True)
    else :
        print("No source file downloaded (none missing locally)"
              , end='\n', file=sys.stderr, flush=True)
    ##############################


    ##############################
    # extract compressed archive #
    ##############################
    if sum([os.path.isfile(os.path.join( root_dir, foldername))
            | os.path.isdir(os.path.join( root_dir, foldername))
            for foldername in foldernames.values()]) != len(foldernames) :
        # if any expected folder is missing locally
        missing_foldernames = dict([foldername for foldername in foldernames.items()
                                    if not os.path.isdir(os.path.join(root_dir, foldername[1]))])
        for i, filename in enumerate(missing_foldernames.items()) :
            print("Extracting sub-folder #" + str(i+1) + "/" + str(len(missing_foldernames.items())) +
                  " (" + '.'.join(filename[0].split('.')[:-1]) + ") :"
                  , file=sys.stderr, flush=True)
            with RarFile(os.path.join( root_dir, filename[0])) as rar_file :
                with tqdm.tqdm(total=len(rar_file.infolist())
                               , bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}' ) as pbar:
                    for f in rar_file.infolist() :
                        file_fullname = os.path.join( root_dir, f.filename)
                        parent_folder_path = \
                            os.path.realpath(os.path.join(file_fullname, '..'))
                        if not os.path.isdir(parent_folder_path) :
                            os.makedirs(parent_folder_path)

                        if f.file_size > 0 and not os.path.isfile(file_fullname) :
                            try :
                                rar_file_entry = rar_file.read(f)
                                with open(file_fullname, 'wb') as target_file :
                                    target_file.write(rar_file_entry)
                            except BadRarFile as brf :
                                if str(brf).startswith('Failed the read enough data') :
                                    msg = \
                                        'did you add your UNRAR location to the system path ? ' + \
                                        '(requires system restart)'
                                    raise RuntimeError(msg) from brf
                                else:
                                    raise

                        pbar.update(1)
    else :
        print("No compressed archive extracted (none missing locally)"
              , end='\n', file=sys.stderr, flush=True)
    sys.stderr.flush()
    ##############################


    grand_timedelta = \
        datetime.timedelta(seconds=(datetime.datetime.now() - grand_start_time).total_seconds())
    grand_timedelta_str = \
        ':'.join(['{:02d}'.format(int(float(i)))
                  for i in str(grand_timedelta).split(':')[:3]])
    print("completed in " + grand_timedelta_str)


    return


#/////////////////////////////////////////////////////////////////////////////////////










































