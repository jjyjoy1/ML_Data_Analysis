def setup_pipe_and_param_grid_kraken(X):
    """
    Set up the pipeline and parameter grid for kraken metagenomics data.
    This function includes both the original models and new added models:
    - Random Survival Forests and Survival SVM for survival analysis
    - Random Forests and Gradient Boosting for drug response analysis
    """
    # Define parameter ranges
    clf_c = np.logspace(-5, 3, 9)
    l1_ratio = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.])
    skb_k = np.insert(np.linspace(2, 400, num=200, dtype=int), 0, 1)
    sfm_c = np.logspace(-2, 3, 6)  # for kraken data
    
    # Import additional models for survival and classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.svm import FastSurvivalSVM
    
    # Set up pipeline based on analysis type and model_type
    # SURVIVAL ANALYSIS for kraken data
    if analysis == 'surv':
        if args.model_type == 'cnet':
            # Original CoxNet model
            pipe = ExtendedPipeline(
                memory=memory,
                param_routing={'srv1': ['feature_meta']},
                steps=[
                    ('trf0', StandardScaler()),
                    ('srv1', MetaCoxnetSurvivalAnalysis(
                        estimator=CachedExtendedCoxnetSurvivalAnalysis(
                            alpha_min_ratio=0.01, fit_baseline_model=True,
                            max_iter=1000000, memory=memory, n_alphas=100,
                            penalty_factor_meta_col=penalty_factor_meta_col,
                            normalize=False, penalty_factor=None)))])
            param_grid_dict = {'srv1__estimator__l1_ratio': l1_ratio}
        
        elif args.model_type == 'rsf':
            # Random Survival Forest
            pipe = ExtendedPipeline(
                memory=memory,
                steps=[
                    ('trf0', StandardScaler()),
                    ('srv1', RandomSurvivalForest(
                        n_estimators=100,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        n_jobs=-1,
                        random_state=random_seed))
                ])
            param_grid_dict = {
                'srv1__n_estimators': [50, 100, 200],
                'srv1__max_depth': [None, 5, 10, 15],
                'srv1__min_samples_split': [5, 10, 15],
                'srv1__min_samples_leaf': [3, 5, 10]
            }
            
        elif args.model_type == 'svm':
            # Survival SVM
            pipe = ExtendedPipeline(
                memory=memory,
                steps=[
                    ('trf0', StandardScaler()),
                    ('srv1', FastSurvivalSVM(
                        alpha=1.0,
                        rank_ratio=1.0,
                        fit_intercept=True,
                        max_iter=1000,
                        random_state=random_seed))
                ])
            param_grid_dict = {
                'srv1__alpha': np.logspace(-3, 3, 7),
                'srv1__rank_ratio': [0.0, 0.5, 1.0]
            }
        else:
            # Default to CoxNet if model_type isn't recognized
            pipe = ExtendedPipeline(
                memory=memory,
                param_routing={'srv1': ['feature_meta']},
                steps=[
                    ('trf0', StandardScaler()),
                    ('srv1', MetaCoxnetSurvivalAnalysis(
                        estimator=CachedExtendedCoxnetSurvivalAnalysis(
                            alpha_min_ratio=0.01, fit_baseline_model=True,
                            max_iter=1000000, memory=memory, n_alphas=100,
                            penalty_factor_meta_col=penalty_factor_meta_col,
                            normalize=False, penalty_factor=None)))])
            param_grid_dict = {'srv1__estimator__l1_ratio': l1_ratio}
    
    # DRUG RESPONSE MODELS
    elif args.model_type == 'rfe':
        # Original RFE-SVM model
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'clf1': ['feature_meta', 'sample_weight']},
            steps=[
                ('trf0', StandardScaler()),
                ('clf1', ExtendedRFE(
                    estimator=SVC(
                        cache_size=2000, class_weight='balanced',
                        kernel='linear', max_iter=int(1e8),
                        random_state=random_seed),
                    memory=memory, n_features_to_select=None,
                    penalty_factor_meta_col=penalty_factor_meta_col,
                    reducing_step=False, step=1, tune_step_at=None,
                    tuning_step=1))])
        param_grid_dict = {'clf1__estimator__C': clf_c,
                          'clf1__n_features_to_select': skb_k}
    
    elif args.model_type == 'lgr':
        # Original Logistic Regression model
        col_trf_col_grps = get_col_trf_col_grps(
            X, [['^(?!gender_male|age_at_diagnosis|tumor_stage).*$',
                 '^(?:gender_male|age_at_diagnosis|tumor_stage)$']])
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'clf1': ['sample_weight'],
                           'trf0': ['sample_weight']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_weight']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr1': ['sample_weight']},
                            steps=[
                                ('trf0', StandardScaler()),
                                ('slr1', SelectFromModel(
                                    estimator=CachedLogisticRegression(
                                        class_weight='balanced',
                                        max_iter=5000,
                                        memory=memory,
                                        penalty='elasticnet',
                                        random_state=random_seed,
                                        solver='saga'),
                                    max_features=400,
                                    threshold=1e-10))]),
                         col_trf_col_grps[0][0]),
                        ('trf1', ExtendedPipeline(
                            memory=memory,
                            param_routing=None,
                            steps=[('trf0', StandardScaler())]),
                         col_trf_col_grps[0][1])])),
                ('clf1', LogisticRegression(
                    class_weight='balanced',
                    max_iter=5000,
                    penalty='l2',
                    random_state=random_seed,
                    solver='saga'))])
        param_grid_dict = {
            'clf1__C': clf_c,
            'trf0__trf0__slr1__estimator__C': sfm_c,
            'trf0__trf0__slr1__estimator__l1_ratio': l1_ratio}
    
    elif args.model_type == 'rf':
        # Random Forest classifier
        col_trf_col_grps = get_col_trf_col_grps(
            X, [['^(?!gender_male|age_at_diagnosis|tumor_stage).*$',
                 '^(?:gender_male|age_at_diagnosis|tumor_stage)$']])
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'trf0': ['sample_weight']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_weight']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr1': ['sample_weight']},
                            steps=[
                                ('trf0', StandardScaler()),
                                ('slr1', SelectFromModel(
                                    estimator=CachedLogisticRegression(
                                        class_weight='balanced',
                                        max_iter=5000,
                                        memory=memory,
                                        penalty='elasticnet',
                                        random_state=random_seed,
                                        solver='saga'),
                                    max_features=400,
                                    threshold=1e-10))]),
                         col_trf_col_grps[0][0]),
                        ('trf1', ExtendedPipeline(
                            memory=memory,
                            param_routing=None,
                            steps=[('trf0', StandardScaler())]),
                         col_trf_col_grps[0][1])])),
                ('clf1', RandomForestClassifier(
                    class_weight='balanced',
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=random_seed))])
        param_grid_dict = {
            'clf1__n_estimators': [50, 100, 200],
            'clf1__max_depth': [None, 5, 10, 15],
            'clf1__min_samples_split': [2, 5, 10],
            'clf1__min_samples_leaf': [1, 2, 4],
            'trf0__trf0__slr1__estimator__C': [0.1, 1.0],
            'trf0__trf0__slr1__estimator__l1_ratio': [0.5, 0.7, 0.9]
        }
        
    elif args.model_type == 'gb':
        # Gradient Boosting classifier
        col_trf_col_grps = get_col_trf_col_grps(
            X, [['^(?!gender_male|age_at_diagnosis|tumor_stage).*$',
                 '^(?:gender_male|age_at_diagnosis|tumor_stage)$']])
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'trf0': ['sample_weight']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_weight']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr1': ['sample_weight']},
                            steps=[
                                ('trf0', StandardScaler()),
                                ('slr1', SelectFromModel(
                                    estimator=CachedLogisticRegression(
                                        class_weight='balanced',
                                        max_iter=5000,
                                        memory=memory,
                                        penalty='elasticnet',
                                        random_state=random_seed,
                                        solver='saga'),
                                    max_features=400,
                                    threshold=1e-10))]),
                         col_trf_col_grps[0][0]),
                        ('trf1', ExtendedPipeline(
                            memory=memory,
                            param_routing=None,
                            steps=[('trf0', StandardScaler())]),
                         col_trf_col_grps[0][1])])),
                ('clf1', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    subsample=1.0,
                    random_state=random_seed))])
        param_grid_dict = {
            'clf1__n_estimators': [50, 100, 200],
            'clf1__learning_rate': [0.01, 0.1, 0.2],
            'clf1__max_depth': [3, 5, 7],
            'clf1__subsample': [0.8, 1.0],
            'trf0__trf0__slr1__estimator__C': [0.1, 1.0],
            'trf0__trf0__slr1__estimator__l1_ratio': [0.5, 0.9]
        }
    
    # limma for kraken data (default if model type is limma/edger)
    else:
        # Original Limma model
        col_trf_col_grps = get_col_trf_col_grps(
            X, [['^(?!gender_male|age_at_diagnosis|tumor_stage).*$']])
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'clf2': ['sample_weight'],
                           'trf0': ['sample_meta']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_meta']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr0': ['sample_meta']},
                            steps=[
                                ('slr0', Limma(
                                    memory=memory,
                                    robust=True,
                                    trend=True))]),
                         col_trf_col_grps[0][0])])),
                ('trf1', StandardScaler()),
                ('clf2', LogisticRegression(
                    class_weight='balanced',
                    max_iter=5000,
                    penalty='l2',
                    random_state=random_seed,
                    solver='saga'))])
        param_grid_dict = {'clf2__C': clf_c,
                           'trf0__trf0__slr0__k': skb_k}
    
    param_grid = [param_grid_dict.copy()]
    return pipe, param_grid, param_grid_dict
