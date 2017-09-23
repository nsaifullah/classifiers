##Who's Leaving?##
##This is an exploratory piece on a Human Resources dataset from Kaggle on workers leaving a company.
##I use successively more sophisticated learning algorithms to classify workers into "Left" and "Not-Left" using the features available.
##Predictably, sklearn's ensemble methods are fantastic for this question.

##Nikhil Saifullah

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.model_selection import RepeatedKFold

#Sectional Run Flags
alg0 = True
alg1 = False
alg2 = False
alg3 = False

#Proportion of total data for test set
p_test = 0.25

#Total dataset:
df = pd.read_csv('C:/Users/bodhisattva_2/Desktop/MLTraining/HR_comma_sep.csv')
n = df.shape[0]

#Define test and training set:
np.random.seed(123456)
df['test'] = pd.Series(np.random.randint(0, n, n))
df['test'] = df['test'].apply(lambda x: 1 if x >= n*(1-p_test) else 0)
df_test = df[df['test'] == 1]; n_test = df_test.shape[0]
df_train = df[df['test'] != 1]; n_train = df_train.shape[0]

##Preliminary Exploration of Training Set:

#Being paid less positively impacts the likelihood of leaving within each department

h1_df = pd.merge(df_train.groupby(['sales', 'salary']).size().to_frame(), df_train.groupby(['sales', 'salary'])['left'].sum().to_frame(), left_index=True, right_index=True, how='inner')
h1_df['left_prop'] = h1_df['left']/h1_df[0]

#The satisfaction variable is strong, as expected, in the negative direction

h2_result = smf.ols(formula="left ~ sales + satisfaction_level", data=df_train).fit()
#print(h2_result.summary())

#Tenure effect is not nearly as strong as subjective satisfaction in this way, and similar tests reveal that average monthly hours, accident & the promotion variable are even weaker
#(the latter two impact too few workers to matter much in the way of leaving likelihood)

h3_result = smf.ols(formula="left ~ sales + time_spend_company", data=df_train).fit()
#print(h3_result.summary())

#The remaining features have lower Mean to StDev ratios and interquartile ranges than the first two big features
df_train.iloc[:, :-5].describe()

##First Model Approach: Linear Probability Model (OLS Regression) - Use only strongest features to set baseline

if (alg0):
	
	c0_result = smf.ols(formula="left ~ sales + satisfaction_level + salary", data=df_train).fit()
	#print(c0_result.summary())
	#df_train['c0_resid'] = c0_result.wresid

	#The residual plot reveals that much of the critical information in the training set is left out of the model
	#Analyzing correlations with training data, residuals and OLS fit confirms this
	# plt.hist(df_train['c0_resid'])
	# df_train[df_train['c0_resid'] < 0.0]['left'].mean(); df_train[df_train['c0_resid'] >= 0.0]['left'].mean()
	# df_train['c0_resid_left'] = df_train['c0_resid'].apply(lambda x: 1 if x >= 0.25 else 0) #Cutoff point determined from histogram plot
	df_train['c0_in_pred'] = c0_result.fittedvalues
	# plt.hist(df_train['c0_in_pred'])
	# df_train[df_train['c0_in_pred'] < df_train['c0_in_pred'].mean()]['left'].mean(); df_train[df_train['c0_in_pred'] >= df_train['c0_in_pred'].mean()]['left'].mean()
	df_train['c0_left'] = df_train['c0_in_pred'].apply(lambda x: 1 if x >= df_train['c0_in_pred'].mean() else 0)

	#Make a confusion matrix for the results to confirm intuition - Note: Using within-prediction mean as cutoff (instead of 50%) since LPM not scaled to be probability
	dcsn_bndry_0 = 0.5
	
	c0_out_result = pd.Series(c0_result.predict({'sales': df_test['sales'], 'salary': df_test['salary'], 'satisfaction_level': df_test['satisfaction_level']}))
	df_test['c0_out_pred'] = pd.Series(c0_out_result)
	
	df_test['c0_left'] = df_test['c0_out_pred'].apply(lambda x: 1 if x >= dcsn_bndry_0 else 0)
	
	print('\nOLS IN Fit Results \n')

	print('Total Negatives ' + str(df_train[df_train['left'] == 0].shape[0]))
	print('True Negatives ' + str(df_train[np.logical_and(df_train['left'] == 0, df_train['c0_left'] == 0)].shape[0]))
	print('Total Positives ' + str(df_train[df_train['left'] == 1].shape[0]))
	print('True Positives ' + str(df_train[np.logical_and(df_train['left'] == 1, df_train['c0_left'] == 1)].shape[0]) + '\n')
	print('Average Accuracy ' + str(df_train[df_train['left'] == df_train['c0_left']].shape[0]/n_train))
	
	print('\nOLS OOS Fit Results \n')

	print('Total Negatives ' + str(df_test[df_test['left'] == 0].shape[0]))
	print('True Negatives ' + str(df_test[np.logical_and(df_test['left'] == 0, df_test['c0_left'] == 0)].shape[0]))
	print('Total Positives ' + str(df_test[df_test['left'] == 1].shape[0]))
	print('True Positives ' + str(df_test[np.logical_and(df_test['left'] == 1, df_test['c0_left'] == 1)].shape[0]) + '\n')
	print('Average Accuracy ' + str(df_test[df_test['left'] == df_test['c0_left']].shape[0]/n_test))

##Second Model Approach: Logit Probability Model - Perhaps the functional form is misspecified; use similar feature set

if (alg1):
	
	#Same feature set as in OLS with logit model
	c1_result = smf.logit(formula="left ~ sales + satisfaction_level + salary", data=df_train).fit()
	#print(c1_result.summary())
	#Confusion matrix; better fit on negatives than OLS but still awful at positives; similar performance OOS
	dcsn_bndry_1 = 0.5
	
	print('Baseline Logit IN Fit Results - Standard Confusion Matrix w/ 50% Decision Boundary')
	print(c1_result.pred_table(0.5))
	c1_out_result = pd.Series(c1_result.predict({'sales': df_test['sales'], 'salary': df_test['salary'], 'satisfaction_level': df_test['satisfaction_level']}))
	df_test['c1_out_pred'] = pd.Series(c1_out_result)
	
	df_test['c1_left'] = df_test['c1_out_pred'].apply(lambda x: 1 if x >= dcsn_bndry_1 else 0)
	
	print('\nBaseline Logit OOS Fit Results \n')

	print('Total Negatives ' + str(df_test[df_test['left'] == 0].shape[0]))
	print('True Negatives ' + str(df_test[np.logical_and(df_test['left'] == 0, df_test['c1_left'] == 0)].shape[0]))
	print('Total Positives ' + str(df_test[df_test['left'] == 1].shape[0]))
	print('True Positives ' + str(df_test[np.logical_and(df_test['left'] == 1, df_test['c1_left'] == 1)].shape[0]) + '\n')
	print('Average Accuracy ' + str(df_test[df_test['left'] == df_test['c1_left']].shape[0]/n_test))
	
	#Add 'number_project' to logit model
	c1_2_result = smf.logit(formula="left ~ sales + satisfaction_level + salary + number_project", data=df_train).fit()
	#print(c1_2_result.summary())
	#Confusion matrix; modest improvement on positives without much of a change in performance on negatives
	print('Confusion Matrix for Logit with number_project added to X IN Results - Standard Confusion Matrix w/ 50% Decision Boundary')
	print(c1_2_result.pred_table(dcsn_bndry_1))
	c1_2_out_result = pd.Series(c1_2_result.predict({'sales': df_test['sales'], 'salary': df_test['salary'], 'satisfaction_level': df_test['satisfaction_level'], 'number_project': df_test['number_project']}))
	df_test['c1_2_out_pred'] = c1_2_out_result
	df_test['c1_2_left'] = df_test['c1_2_out_pred'].apply(lambda x: 1 if x >= dcsn_bndry_1 else 0)
	
	# print('\nLogit w/ number_project OOS Fit Results \n')

	# print('Total Negatives ' + str(df_test[df_test['left'] == 0].shape[0]))
	# print('True Negatives ' + str(df_test[np.logical_and(df_test['left'] == 0, df_test['c1_2_left'] == 0)].shape[0]))
	# print('Total Positives ' + str(df_test[df_test['left'] == 1].shape[0]))
	# print('True Positives ' + str(df_test[np.logical_and(df_test['left'] == 1, df_test['c1_2_left'] == 1)].shape[0]) + '\n')
	# print('Average Accuracy ' + str(df_test[df_test['left'] == df_test['c1_2_left']].shape[0]/n_test))
	
	#4-Fold CV done twice each fold to check average OOS accuracy reported above
	
	rkf = RepeatedKFold(n_splits=4, n_repeats=2, random_state=12883823)
	cv_score_list = []
	for (train, test) in rkf.split(df):
		c1_2_result = smf.logit(formula="left ~ sales + satisfaction_level + salary + number_project", data=df.iloc[train]).fit()
		df_test_cv = df.iloc[test]
		c1_2_out_result = pd.Series(c1_2_result.predict({'sales': df_test_cv['sales'], 'salary': df_test_cv['salary'], 'satisfaction_level': df_test_cv['satisfaction_level'], 'number_project': df_test_cv['number_project']}))
		df_test_cv['c1_2_out_pred'] = c1_2_out_result
		df_test_cv['c1_2_left'] = df_test_cv['c1_2_out_pred'].apply(lambda x: 1 if x >= dcsn_bndry_1 else 0)
		cv_score_list.append(df_test_cv[df_test_cv['left'] == df_test_cv['c1_2_left']].shape[0]/df_test_cv.shape[0])
	print('\nAverage CV Accuracy ' + str(sum(cv_score_list)/len(cv_score_list)))
	del df_test_cv

##Third Model Approach: Random Forest (RF) - Does RF find the same features my descriptive work did and if features interact in non-linear ways

if (alg2):
	df_rf = pd.get_dummies(df, columns=['sales', 'salary'])
	df_train_rf = pd.get_dummies(df_train, columns=['sales', 'salary'])
	df_test_rf = pd.get_dummies(df_test, columns=['sales', 'salary'])
	X = df_train_rf[['satisfaction_level', 'salary_low', 'salary_medium', 'salary_high', 'number_project'] + [v for v in df_train_rf.columns if re.search('sales', v) is not None]]
	X_test = df_test_rf[['satisfaction_level', 'salary_low', 'salary_medium', 'salary_high', 'number_project'] + [v for v in df_test_rf.columns if re.search('sales', v) is not None]]
	c2_result = rfc(max_depth=4, random_state=1234)
	c2_result.fit(X, df_train_rf['left'])

	#Overall  fit is markedly better than previous attempts
	#Decision boundary is implicitly 50% in score() method
	print('\nRandom Forest OOS Results\n')
	print('Average Accuracy ' + str(c2_result.score(X_test, df_test_rf['left'])))
	c2_result.feature_importances_ #number_project is the second most important factor!
	
	#4-Fold CV done twice each fold to check average OOS accuracy reported above
	
	rkf = RepeatedKFold(n_splits=4, n_repeats=2, random_state=12883823)
	cv_score_list = []
	for (train, test) in rkf.split(df_rf[['satisfaction_level', 'salary_low', 'salary_medium', 'salary_high', 'number_project'] + [v for v in df_rf.columns if re.search('sales', v) is not None]]):
		c2_result = rfc(max_depth=4, random_state=1234)
		c2_result.fit(df_rf.iloc[train], df_rf.iloc[train]['left'])
		cv_score_list.append(c2_result.score(df_rf.iloc[test], df_rf.iloc[test]['left']))
	print('\nAverage CV Accuracy ' + str(sum(cv_score_list)/len(cv_score_list)))

##Fourth Model Approach: Boosted Trees (BT) - How does BT compare with RF and the rest of my attempts out-of-the-box? What tuning may be needed?

if (alg3):
	df_bst = pd.get_dummies(df, columns=['sales', 'salary'])
	df_train_bst = pd.get_dummies(df_train, columns=['sales', 'salary'])
	df_test_bst = pd.get_dummies(df_test, columns=['sales', 'salary'])
	X = df_train_bst[['satisfaction_level', 'salary_low', 'salary_medium', 'salary_high', 'number_project'] + [v for v in df_train_bst.columns if re.search('sales', v) is not None]]
	X_test = df_test_bst[['satisfaction_level', 'salary_low', 'salary_medium', 'salary_high', 'number_project'] + [v for v in df_test_bst.columns if re.search('sales', v) is not None]]
	c3_result = abc(random_state=1234)
	c3_result.fit(X, df_train_bst['left'])

	#Overall  fit is markedly better than previous attempts
	#Decision boundary is implicitly 50% in score() method
	print('\nRandom Forest OOS Results\n')
	print('Average Accuracy ' + str(c3_result.score(X_test, df_test_bst['left'])))
	
	#4-Fold CV done twice each fold to check average OOS accuracy reported above
	
	rkf = RepeatedKFold(n_splits=4, n_repeats=2, random_state=12883823)
	cv_score_list = []
	for (train, test) in rkf.split(df_bst[['satisfaction_level', 'salary_low', 'salary_medium', 'salary_high', 'number_project'] + [v for v in df_bst.columns if re.search('sales', v) is not None]]):
		c3_result = abc(random_state=1234)
		c3_result.fit(df_bst.iloc[train], df_bst.iloc[train]['left'])
		cv_score_list.append(c3_result.score(df_bst.iloc[test], df_bst.iloc[test]['left']))
	print(cv_score_list)
	print('\nAverage CV Accuracy ' + str(sum(cv_score_list)/len(cv_score_list)))

print('End of File')
