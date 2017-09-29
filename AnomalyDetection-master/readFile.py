#xls to csv conversion

import pandas as pd

data_xls = pd.read_excel('variance_data_excel.xls', 'Sheet1', index_col=None)
data_xls.to_csv('out.csv', encoding='utf-8')
