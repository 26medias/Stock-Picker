import neptune

opt_name = 'sandbox'

if opt_name=='sandbox':
	neptune.init('leap-forward/sandbox')
	neptune.create_experiment(name='minimal_example')

	# log some metrics

	for i in range(100):
	    neptune.log_metric('loss', 0.95**i)

	neptune.log_metric('AUC', 0.96)
elif opt_name=='':
	import pandas as pd
	import lightgbm as lgb
	from sklearn.model_selection import train_test_split

	SEARCH_PARAMS = {'learning_rate': 0.4,
	                 'max_depth': 15,
	                 'num_leaves': 20,
	                 'feature_fraction': 0.8,
	                 'subsample': 0.2}

	data = pd.read_csv('../data/train.csv', nrows=10000)
	X = data.drop(['ID_code', 'target'], axis=1)
	y = data['target']
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1234)

	train_data = lgb.Dataset(X_train, label=y_train)
	valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

	params = {'objective': 'binary',
	          'metric': 'auc',
	          **SEARCH_PARAMS}

	model = lgb.train(params, train_data,
	                  num_boost_round=300,
	                  early_stopping_rounds=30,
	                  valid_sets=[valid_data],
	                  valid_names=['valid'])

	score = model.best_score['valid']['auc']
	print('validation AUC:', score)