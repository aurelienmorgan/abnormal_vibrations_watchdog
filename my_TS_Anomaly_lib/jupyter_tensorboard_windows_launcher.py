from tensorboard import version, program, default
import logging
import sys, os,  re

# WARNING: Logging before flag parsing goes to stderr.
logging._warn_preinit_stderr = 0



events_filename_pattern = re.compile('events\.out\.tfevents\..*\.MSI')
host_port_pattern_str = '(?:http.*://)?(?P<host>[^:/ ]+).?(?P<port>[0-9]*).*'

class JupyterTensorboardWindows:
    """
    Tensorboard convenience launching class/method
    (instead of relying on the buggy
    "%tensorboard --logdir {log_dir}" command on MS Windows)
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self):
        # Remove http messages
        dummy = logging.getLogger('werkzeug').setLevel(logging.ERROR)

        # address the tensorboard "unable to get first event timestamp for run" bug
        events_folders = [ root
                           for root, dirs, files in os.walk(self.dir_path)
                           for name in files
                           if events_filename_pattern.match(name) ]
        #print(str(events_folders))

        # Start tensorboard server
        self.tb = program.TensorBoard(
            default.get_plugins(), program.get_default_assets_zip_provider())
        self.tb.configure(argv=[None, '--logdir', self.dir_path])
        url = self.tb.launch()
        sys.stdout.write('TensorBoard %s at %s [ %s ]\n' %
                         (version.VERSION, url
                          , "http://localhost:" + re.search(host_port_pattern_str, url).group('port')
                         ))



# REMARK : @see 'https://github.com/tensorflow/tensorflow/issues/9512'
#
# TensorBoard doesn't like it when we write multiple event files from separate runs
# in the same directory.
""" ERROR MESSAGE DISPLAYED AT STARTUP =>

Found more than one graph event per run, or there was a metagraph containing a graph_def,
as well as one or more graph events.  Overwriting the graph with the newest event.

Found more than one metagraph event per run. Overwriting the metagraph with the newest event.
"""
# If you see this message, you have a directory with events for more than one run
# You didn't use a different subdirectory for every run (new hyperparameters = new subdirectory).