import math


#/////////////////////////////////////////////////////////////////////////////////////


millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(n):
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
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


#/////////////////////////////////////////////////////////////////////////////////////





































