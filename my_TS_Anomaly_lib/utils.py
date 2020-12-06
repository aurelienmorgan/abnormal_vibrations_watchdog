import io, sys, datetime
import math


#/////////////////////////////////////////////////////////////////////////////////////


millnames = ['',' Thousand',' Million',' Billion',' Trillion']
short_millnames = ['','k','M','B','T']

def millify(n, use_short_millnames: bool = False):
    """
    Parameters :
        - n (float) :
            large number to be formatted

    Results :
        - (str) :
            Human-readable large numbers string
            pretty-formatted.
    """

    n = float(n)
    names = millnames if (not use_short_millnames) else short_millnames
    millidx = max(0,min(len(names)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), names[millidx])


#/////////////////////////////////////////////////////////////////////////////////////


import io

def print_timedelta(
    start_datetime: datetime.datetime
    , end_datetime: datetime.datetime = None
    , file: io.TextIOBase = sys.stdout
) -> None :
    """
    Pretty-prints the durations

    Parameters :
        - start_datetime (datetime.datetime) :
            datetime value from which the time-difference
            shall be evaluated
        - end_datetime (datetime.datetime) :
            datetime value up to which the time-difference
            shall be evaluated (defaults to 'now')
        - file (io.TextIOBase) :
            character and line based interface
            to stream I/O to which the message shall be sent.

    Results :
        - N.A.
    """

    end_datetime = (datetime.datetime.now() if end_datetime == None else end_datetime)
    timedelta = \
        datetime.timedelta(
            seconds = (end_datetime - start_datetime).total_seconds()
        )
    timedelta_str = \
        ':'.join(['{:02d}'.format(int(float(i)))
                  for i in str(timedelta).split(':')[:3]])
    print("completed in " + timedelta_str, file = file)

    return


#/////////////////////////////////////////////////////////////////////////////////////


from string import Formatter
from datetime import timedelta

def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """
    Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can 
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the  
    default, which is a datetime.timedelta object.  Valid inputtype strings: 
        's', 'seconds', 
        'm', 'minutes', 
        'h', 'hours', 
        'd', 'days', 
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


#/////////////////////////////////////////////////////////////////////////////////////


def df_highlight_row(df, row_index) :
    def row_index_bold_style(row_index: int) :
        """
        return a pandas.DataFrame 'style apply' lambda function.
        If row index == row_index, the format of the cells
        is made orange background, 40% opac & bold-weight font.
        """
        def new_best_row_bold_style(row):
            if row.name == row_index :
                styles = {col: "font-weight: bold; background-color: orange; opacity: 0.4" for col in row.index}
            else :
                styles = {col: "" for col in row.index}
            return styles
        return new_best_row_bold_style

    return df.style.apply(
        lambda df_row: row_index_bold_style(row_index)(df_row)
        , axis=1)


#/////////////////////////////////////////////////////////////////////////////////////


import json, os, re
import ipykernel
import requests
from requests.compat import urljoin
from notebook.notebookapp import list_running_servers

def get_notebook_fullname():
    """
    Return the full path of the jupyter notebook.
    @see https://github.com/jupyter/notebook/issues/1000#issuecomment-359875246
    """
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if not isinstance(nn, str) and nn['kernel']['id'] == kernel_id:
                relative_path = nn['notebook']['path']
                return os.path.realpath(os.path.join(ss['notebook_dir'], relative_path))


#/////////////////////////////////////////////////////////////////////////////////////































































