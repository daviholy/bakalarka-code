import matplotlib

matplotlib.use('TkAgg')

# csv_file = 'https://vincentarelbundock.github.io/Rdatasets/csv/MASS/survey.csv'

# # Reading the CSV file from the URL
# df_s = pd.read_csv(csv_file, index_col=0)

# # Checking the data quickly (first 5 rows):
# df_s.head()

# # matplotlib.use('QtAgg')
# pd.plotting.scatter_matrix(df_s.iloc[:, 1:9])

import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()