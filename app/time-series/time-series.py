from app.data_utils import create_df
import matplotlib.pyplot as plt

data = create_df('../../files/NYPD_short.csv')
minDate = data["CMPLNT_FR"].min()
maxDate = data["CMPLNT_FR"].max()
data = data.set_index(["CMPLNT_FR"])
crime_dates = data.loc["2017" : "2018"].index.value_counts().to_frame('count')

monthly_crimes = crime_dates['count'].resample('M').sum().to_frame('count')

plt.plot(monthly_crimes.index.values, monthly_crimes['count'])
plt.show()