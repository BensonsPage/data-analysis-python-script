# Import data Analysis packages
import pandas as pd
import numpy as np
# Import plotting packages

import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalysis:
	def __init__(self):
		# Main Data Path
		benign_phish_data_path = "benign_phish_links_features.csv"

		# read Data
		self.benign_phish_data = pd.read_csv(benign_phish_data_path)
		# Encoding 'File' as label benign(1) & phish(0), naming the field as target
		self.benign_phish_data.File.replace({'tt-data/phish_links': 0, 'tt-data/benign_links_tiny': 1,
											 'tt-data/benign_links': 1,
											 'tt-data/phish_links_tiny': 0}, inplace=True)
		self.benign_phish_data.rename(columns={'File': 'target'}, inplace=True)
		# Normalize Data Types
		self.benign_phish_data.replace(True, 1, inplace=True)
		self.benign_phish_data.replace(False, 0, inplace=True)

	def clean_data(self):
		# Print Data Types
		print(self.benign_phish_data.dtypes)
		# Print Columns
		print(self.benign_phish_data.columns)
		# Print shape before cleaning
		print("Shape before cleaning: ", self.benign_phish_data.shape)
		# # Drop unnamed columns
		# self.benign_phish_data.drop(columns = "Unnamed: 0", inplace = True)

		# # Slice data, get rid of some last rows when testing.
		# self.benign_phish_data.drop(self.benign_phish_data.shape(2000).index, inplace=True)
		# print("Shape after slicing: ", self.benign_phish_data.shape)

		# Drop unnecessary columns. i.e, columns with constant values
		drop_candidates = ['urlHasPortInString', 'has_ip']
		self.benign_phish_data.drop(columns=drop_candidates, inplace=True)

		# Print shape after removing unnecessary columns
		print("Shape after removing unnecessary columns: ", self.benign_phish_data.shape)

		# Drop null rows (Rows with Missing Data)
		self.benign_phish_data = self.benign_phish_data.dropna()

		# Print shape after removing rows with null/empty
		print("Shape after removing null/empty: ", self.benign_phish_data.shape)

		try:
			# Drop Duplicates
			self.benign_phish_data = DataAnalysisObj.remove_duplicates()

		except NameError:
			print("Can't remove duplicates")
		finally:
			# Print shape after removing duplicates
			print("Shape after removing duplicates: ", self.benign_phish_data.shape)

		self.benign_phish_data = DataAnalysisObj.remove_outliers()
		# Print shape after removing outliers
		print("Shape after removing outliers: ", self.benign_phish_data.shape)

		# Export Clean Data
		self.benign_phish_data.to_csv('benign_phish_links_data_processed.csv')

	def remove_outliers(self):
		# Remove Outliers using IQR (Inter Quartile Range)
		# Calculate the upper and lower limits

		columns_with_outliers = ['numberOfPeriods']

		for var in columns_with_outliers:
			q1 = self.benign_phish_data[var].quantile(0.25)
			q3 = self.benign_phish_data[var].quantile(0.75)
			iqr = q3 - q1
			lower = q1 - 1.2*iqr
			upper = q3 + 1.2*iqr

			# Create arrays of Boolean values indicating the outlier rows
			upper_array = np.where(self.benign_phish_data[var] >= upper)[0]
			lower_array = np.where(self.benign_phish_data[var] <= lower)[0]

			# Removing the outliers
			if any(upper_array) != 0:
				print("Upper OUTLIERS --- ", upper_array)
				self.benign_phish_data.drop(index=upper_array, axis=1, inplace=True)
			if any(lower_array) != 0:
				print("Lower OUTLIERS --- ", lower_array)
				self.benign_phish_data.drop(index=lower_array, axis=1, inplace=True)

		self.benign_phish_data.to_csv('benign_phish_data_outliers_free.csv')

		return self.benign_phish_data

	def remove_duplicates(self):
		# # Drop Duplicates
		self.benign_phish_data.drop_duplicates(subset=['scriptLength'], keep='first', inplace=True, ignore_index=True)
		return self.benign_phish_data

	def explore_data(self):
		# Plotting data shape
		print("Shape of plotting data: ", self.benign_phish_data.shape)
		print(self.benign_phish_data.head(10))

		# Pearson Correlation Heatmap (Shows strength of relationship between variable)
		plt.figure(figsize=(14, 6))
		sns.set(font_scale=0.5)
		sns.heatmap(self.benign_phish_data.corr(), annot=True, cmap="YlGnBu")
		plt.show()

		# Distribution plot
		sns.displot(self.benign_phish_data, x="target", y="entropy", hue="hasHttp")
		plt.show()

		# Box plot for identifying outliers, showing distribution between numeric data
		boxplot_data = self.benign_phish_data.corr()
		ax = sns.boxplot(boxplot_data)
		ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
		plt.show()

		# Scatter Plot
		sns.scatterplot(data=self.benign_phish_data, x="hasHttps", y="target")
		plt.show()

		# Violin Plot
		sns.violinplot(data=self.benign_phish_data, x="numImages", y="numLinks", hue="target")
		plt.show()


DataAnalysisObj = DataAnalysis()
DataAnalysisObj.clean_data()
DataAnalysisObj.explore_data()
