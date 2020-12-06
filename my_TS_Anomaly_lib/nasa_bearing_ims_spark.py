import datetime, re

from pyspark.sql.types import StructType, StructField \
                              , StringType, FloatType, DoubleType \
                              , IntegerType, TimestampType
from pyspark.sql import SparkSession, functions as F


#/////////////////////////////////////////////////////////////////////////////////////


schema = StructType(
    [
      StructField("_id", StructType([StructField("oid", StringType(), False)]), False)
      , StructField("sensor_1", DoubleType(), False)
      , StructField("sensor_2", DoubleType(), False)
      , StructField("sensor_3", DoubleType(), False)
      , StructField("sensor_4", DoubleType(), False)
      , StructField("sensor_5", DoubleType(), True)
      , StructField("sensor_6", DoubleType(), True)
      , StructField("sensor_7", DoubleType(), True)
      , StructField("sensor_8", DoubleType(), True)
      , StructField("step", IntegerType(), False)
      , StructField("test_id", IntegerType(), False)
      , StructField("timestamp", TimestampType(), False)
    ]
)


#/////////////////////////////////////////////////////////////////////////////////////


def test_plot(data_df, test_id, ax) -> None :
    """
    Plots a sensor measurements timeline
    (can be '.mean().avg()' pre-processed).

    Parameters :
        - data_df (pandas.DataFrame) :
            one datapoint per row dataset.
        - test_id (int) :
            ID of the test to be plotted.
        - ax (matplotlib.pyplot.axes.Axes) :
            subplot axes object to plot with.

    Results :
        - N.A.
    """

    assert set(['test_id', 'sensor_1', 'relative_timestamp']).issubset(data_df.columns) \
           , 'input dataframe structure exception'
    assert test_id in range(1, 4) \
           , 'test_id {{test_id}} invalid'.format(test_id)

    df = data_df.loc[(data_df['test_id'] == test_id)
                     , 'sensor_1':'relative_timestamp']
    for sensor_name, curve_color, curve_thickness \
    in zip(['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4'
            , 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8']
           , ['blue', 'red', 'green', 'black'
              , 'yellow', 'orange', 'purple', 'pink']
           , [1, 1, 1, 1, 1, 1, 1, .3]) :
        ax.plot(
            df['relative_timestamp']
            , list(df[sensor_name]), label = sensor_name
            , color=curve_color, animated = True, linewidth=curve_thickness)
    
    ax.set_title("test #" + str(test_id), y=.93, x=.85)
    ax.margins(x=.01, y=.01)
    x_tick_location = range(0, int(max(ax.get_xticks()))
                            , int(int(max(ax.get_xticks()))/len(ax.get_xticks())))
    x_tick_labels = [
        datetime.timedelta(microseconds=pos/1000).days
        for pos in x_tick_location
    ]
    ax.set_xticks(x_tick_location, minor=False)
    ax.set_xticklabels(x_tick_labels, minor=False)
    ax.set_xlabel('days')


#/////////////////////////////////////////////////////////////////////////////////////


def get_sensors_capped_sql_statement(
    spark: SparkSession
    , test_id: int
    , save_last_days_count: int
) -> str :
    """
    function returning the sql statement to be queried
    in order to get measurement values
    from all the sensors for a given test up to a certain time,
    specified by the number of days to be omitted from the endpoint.

    Parameters :
        - spark (SparkSession) :
            The spark session over which the query is to be run
            (need-to-know : custom function registration purpose).
        - test_id (int) :
            ID of the test to be considered (1-3).
        - save_last_days_count (int) :
            How many days of measurements shall be excluded
            (counting backwards from the end of the test)

    Results :
        - (str) :
            SQL statement
    """

    def datetime_diff(time2: datetime.datetime, time1: datetime.datetime) :
        """
        function returning the amount of seconds between two datetimes.
        """
        diff = time2 - time1
        return diff.total_seconds()

    spark.udf.register(name = "datetime_diff"
                       , f = F.udf(lambda x, y: datetime_diff(x,y)
                                   , returnType=DoubleType()))

    sql_statement = \
        """
        |SELECT
        |    tab.timestamp
        |    , tab.sensor_1
        |    , tab.sensor_2
        |    , tab.sensor_3
        |    , tab.sensor_4
        |FROM
        |    (
        |        SELECT
        |            tmp.timestamp
        |            , datetime_diff(
        |                CAST((SELECT max(timestamp) as max_timestamp
        |                      FROM test_""" + str(test_id) + """_agg_tab) AS timestamp)
        |                , tmp.timestamp
        |            ) AS seconds_up_to_end
        |            , tmp.sensor_1
        |            , tmp.sensor_2
        |            , tmp.sensor_3
        |            , tmp.sensor_4
        |        FROM
        |            test_""" + str(test_id) + """_agg_tab tmp
        |    ) tab
        |WHERE
        |    tab.seconds_up_to_end > (""" + str(save_last_days_count) + """*24*60*60) -- seconds in 'save_last_days' days
        |ORDER BY
        |    tab.timestamp ASC
        """

    def strip_margin(text):
        return re.sub('\n[ \t]*\|', '\n', text)

    return strip_margin(sql_statement)


#/////////////////////////////////////////////////////////////////////////////////////


@F.udf(returnType = DoubleType())
def vector_row_element(v, i) :
    """element 'i' from a Spark row object"""
    try: return float(v[i])
    except ValueError: return None


#/////////////////////////////////////////////////////////////////////////////////////


















































