import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/omer/teknofest_iha_2_gorev/install/teknofest_iha'
