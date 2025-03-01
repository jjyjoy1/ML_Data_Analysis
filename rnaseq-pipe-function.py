def setup_pipe_and_param_grid_rnaseq(X):
    """
    Set up the pipeline and parameter grid for RNA-seq data (ENSG patterns).
    This function is extracted from the original setup_pipe_and_param_grid function
    but only contains the parts relevant to RNA-seq data processing.
    """
    # Define parameter ranges
    clf_c = np.logspace(-5, 3, 9)
    l1_ratio = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.])
    skb_k = np.insert(np.linspace(2, 400, num=200, dtype=int), 0, 1)
    sfm_c = np.logspace(-2, 1, 4)  # for RNA-seq data
    
    # Import additional models for survival and classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.svm import FastSurvivalSVM
    
    # Set up pipeline based on analysis type and model_type
    # SURVIVAL ANALYSIS for RNA-seq data
    if analysis == 'surv':
        # Extract RNA-seq specific columns (ENSG patterns)
        col_trf_col_grps = get_col_trf_col_grps(X, [['^ENSG.+$']])
        
        if args.model_type == 'cnet':
            # CoxNet model for RNA-seq
            pipe = ExtendedPipeline(
                memory=memory,
                param_routing={'srv2': ['feature_meta'],
                               'trf0': ['sample_meta']},
                steps=[
                    ('trf0', ExtendedColumnTransformer(
                        n_jobs=1,
                        param_routing={'trf0': ['sample_meta']},
                        remainder='passthrough',
                        transformers=[
                            ('trf0', ExtendedPipeline(
                                memory=memory,
                                param_routing={'slr0': ['sample_meta'],
                                               'trf1': ['sample_meta']},
                                steps=[
                                    ('slr0', EdgeRFilterByExpr(
                                        is_classif=False)),
                                    ('trf1', EdgeRTMMLogCPM(
                                        prior_count=1))]),
                             col_trf_col_grps[0][0])])),
                    ('trf1', StandardScaler()),
                    ('srv2', MetaCoxnetSurvivalAnalysis(
                        estimator=CachedExtendedCoxnetSurvivalAnalysis(
                            alpha_min_ratio=0.01, fit_baseline_model=True,
                            max_iter=1000000, memory=memory, n_alphas=100,
                            penalty_factor_meta_col=penalty_factor_meta_col,
                            normalize=False, penalty_factor=None)))])
            param_grid_dict = {'srv2__estimator__l1_ratio': l1_ratio}
            
        elif args.model_type == 'rsf':
            # Random Survival Forest for RNA-seq
            pipe = ExtendedPipeline(
                memory=memory,
                param_routing={'trf0': ['sample_meta']},
                steps=[
                    ('trf0', ExtendedColumnTransformer(
                        n_jobs=1,
                        param_routing={'trf0': ['sample_meta']},
                        remainder='passthrough',
                        transformers=[
                            ('trf0', ExtendedPipeline(
                                memory=memory,
                                param_routing={'slr0': ['sample_meta'],
                                               'trf1': ['sample_meta']},
                                steps=[
                                    ('slr0', EdgeRFilterByExpr(
                                        is_classif=False)),
                                    ('trf1', EdgeRTMMLogCPM(
                                        prior_count=1))]),
                             col_trf_col_grps[0][0])])),
                    ('trf1', StandardScaler()),
                    ('srv2', RandomSurvivalForest(
                        n_estimators=100,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        n_jobs=-1,
                        random_state=random_seed))
                ])
            param_grid_dict = {
                'srv2__n_estimators': [50, 100, 200],
                'srv2__max_depth': [None, 5, 10, 15],
                'srv2__min_samples_split': [5, 10, 15],
                'srv2__min_samples_leaf': [3, 5, 10]
            }
            
        elif args.model_type == 'svm':
            # Survival SVM for RNA-seq
            pipe = ExtendedPipeline(
                memory=memory,
                param_routing={'trf0': ['sample_meta']},
                steps=[
                    ('trf0', ExtendedColumnTransformer(
                        n_jobs=1,
                        param_routing={'trf0': ['sample_meta']},
                        remainder='passthrough',
                        transformers=[
                            ('trf0', ExtendedPipeline(
                                memory=memory,
                                param_routing={'slr0': ['sample_meta'],
                                               'trf1': ['sample_meta']},
                                steps=[
                                    ('slr0', EdgeRFilterByExpr(
                                        is_classif=False)),
                                    ('trf1', EdgeRTMMLogCPM(
                                        prior_count=1))]),
                             col_trf_col_grps[0][0])])),
                    ('trf1', StandardScaler()),
                    ('srv2', FastSurvivalSVM(
                        alpha=1.0,
                        rank_ratio=1.0,
                        fit_intercept=True,
                        max_iter=1000,
                        random_state=random_seed))
                ])
            param_grid_dict = {
                'srv2__alpha': np.logspace(-3, 3, 7),
                'srv2__rank_ratio': [0.0, 0.5, 1.0]
            }
        else:
            # Default to CoxNet if model_type isn't recognized
            pipe = ExtendedPipeline(
                memory=memory,
                param_routing={'srv2': ['feature_meta'],
                               'trf0': ['sample_meta']},
                steps=[
                    ('trf0', ExtendedColumnTransformer(
                        n_jobs=1,
                        param_routing={'trf0': ['sample_meta']},
                        remainder='passthrough',
                        transformers=[
                            ('trf0', ExtendedPipeline(
                                memory=memory,
                                param_routing={'slr0': ['sample_meta'],
                                               'trf1': ['sample_meta']},
                                steps=[
                                    ('slr0', EdgeRFilterByExpr(
                                        is_classif=False)),
                                    ('trf1', EdgeRTMMLogCPM(
                                        prior_count=1))]),
                             col_trf_col_grps[0][0])])),
                    ('trf1', StandardScaler()),
                    ('srv2', MetaCoxnetSurvivalAnalysis(
                        estimator=CachedExtendedCoxnetSurvivalAnalysis(
                            alpha_min_ratio=0.01, fit_baseline_model=True,
                            max_iter=1000000, memory=memory, n_alphas=100,
                            penalty_factor_meta_col=penalty_factor_meta_col,
                            normalize=False, penalty_factor=None)))])
            param_grid_dict = {'srv2__estimator__l1_ratio': l1_ratio}
    
    # DRUG RESPONSE MODELS for RNA-seq data
    elif args.model_type == 'rfe':
        # Extract RNA-seq specific columns (ENSG patterns)
        col_trf_col_grps = get_col_trf_col_grps(X, [['^ENSG.+$']])
        
        # RFE-SVM for RNA-seq
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'clf2': ['feature_meta', 'sample_weight'],
                           'trf0': ['sample_meta']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_meta']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr0': ['sample_meta'],
                                           'trf1': ['sample_meta']},
                            steps=[
                                ('slr0', EdgeRFilterByExpr(
                                    is_classif=True)),
                                ('trf1', EdgeRTMMLogCPM(
                                    prior_count=1))]),
                         col_trf_col_grps[0][0])])),
                ('trf1', StandardScaler()),
                ('clf2', ExtendedRFE(
                    estimator=SVC(
                        cache_size=2000, class_weight='balanced',
                        kernel='linear', max_iter=int(1e8),
                        random_state=random_seed),
                    memory=memory, n_features_to_select=None,
                    penalty_factor_meta_col=penalty_factor_meta_col,
                    reducing_step=True, step=0.05, tune_step_at=1300,
                    tuning_step=1))])
        param_grid_dict = {'clf2__estimator__C': clf_c,
                           'clf2__n_features_to_select': skb_k}
    
    elif args.model_type == 'lgr':
        # Extract RNA-seq specific columns (ENSG patterns)
        col_trf_col_grps = get_col_trf_col_grps(X, [['^ENSG.+$']])
        
        # Logistic Regression for RNA-seq
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'clf1': ['sample_weight'],
                           'trf0': ['sample_meta', 'sample_weight']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_meta',
                                            'sample_weight']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr0': ['sample_meta'],
                                           'slr3': ['sample_weight'],
                                           'trf1': ['sample_meta']},
                            steps=[
                                ('slr0', EdgeRFilterByExpr(
                                    is_classif=True)),
                                ('trf1', EdgeRTMMLogCPM(
                                    prior_count=1)),
                                ('trf2', StandardScaler()),
                                ('slr3', SelectFromModel(
                                    estimator=CachedLogisticRegression(
                                        class_weight='balanced',
                                        max_iter=5000,
                                        memory=memory,
                                        penalty='elasticnet',
                                        random_state=random_seed,
                                        solver='saga'),
                                    max_features=400,
                                    threshold=1e-10))]),
                         col_trf_col_grps[0][0])])),
                ('clf1', LogisticRegression(
                    class_weight='balanced',
                    max_iter=5000,
                    penalty='l2',
                    random_state=random_seed,
                    solver='saga'))])
        param_grid_dict = {
            'clf1__C': clf_c,
            'trf0__trf0__slr3__estimator__C': sfm_c,
            'trf0__trf0__slr3__estimator__l1_ratio': l1_ratio}
    
    elif args.model_type == 'rf':
        # Extract RNA-seq specific columns (ENSG patterns)
        col_trf_col_grps = get_col_trf_col_grps(X, [['^ENSG.+$']])
        
        # Random Forest for RNA-seq
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'trf0': ['sample_meta', 'sample_weight']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_meta',
                                            'sample_weight']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr0': ['sample_meta'],
                                           'slr3': ['sample_weight'],
                                           'trf1': ['sample_meta']},
                            steps=[
                                ('slr0', EdgeRFilterByExpr(
                                    is_classif=True)),
                                ('trf1', EdgeRTMMLogCPM(
                                    prior_count=1)),
                                ('trf2', StandardScaler()),
                                ('slr3', SelectFromModel(
                                    estimator=CachedLogisticRegression(
                                        class_weight='balanced',
                                        max_iter=5000,
                                        memory=memory,
                                        penalty='elasticnet',
                                        random_state=random_seed,
                                        solver='saga'),
                                    max_features=400,
                                    threshold=1e-10))]),
                         col_trf_col_grps[0][0])])),
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
            'trf0__trf0__slr3__estimator__C': [0.1, 1.0],
            'trf0__trf0__slr3__estimator__l1_ratio': [0.5, 0.7, 0.9]
        }
    
    elif args.model_type == 'gb':
        # Extract RNA-seq specific columns (ENSG patterns)
        col_trf_col_grps = get_col_trf_col_grps(X, [['^ENSG.+$']])
        
        # Gradient Boosting for RNA-seq
        pipe = ExtendedPipeline(
            memory=memory,
            param_routing={'trf0': ['sample_meta', 'sample_weight']},
            steps=[
                ('trf0', ExtendedColumnTransformer(
                    n_jobs=1,
                    param_routing={'trf0': ['sample_meta',
                                            'sample_weight']},
                    remainder='passthrough',
                    transformers=[
                        ('trf0', ExtendedPipeline(
                            memory=memory,
                            param_routing={'slr0': ['sample_meta'],
                                           'slr3': ['sample_weight'],
                                           'trf1': ['sample_meta']},
                            steps=[
                                ('slr0', EdgeRFilterByExpr(
                                    is_classif=True)),
                                ('trf1', EdgeRTMMLogCPM(
                                    prior_count=1)),
                                ('trf2', StandardScaler()),
                                ('slr3', SelectFromModel(
                                    estimator=CachedLogisticRegression(
                                        class_weight='balanced',
                                        max_iter=5000,
                                        memory=memory,
                                        penalty='elasticnet',
                                        random_state=random_seed,
                                        solver='saga'),
                                    max_features=400,
                                    threshold=1e-10))]),
                         col_trf_col_grps[0][0])])),
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
            'trf0__trf0__slr3__estimator__C': [0.1, 1.0],
            'trf0__trf0__slr3__estimator__l1_ratio': [0.5, 0.7, 0.9]
        }
    
    else:
        # Extract RNA-seq specific columns (ENSG patterns)
        col_trf_col_grps = get_col_trf_col_grps(X, [['^ENSG.+$']])
        
        # EdgeR model for RNA-seq as default
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
                            param_routing={'slr0': ['sample_meta'],
                                           'slr1': ['sample_meta']},
                            steps=[
                                ('slr0', EdgeRFilterByExpr(
                                    is_classif=True)),
                                ('slr1', EdgeR(
                                    memory=memory,
                                    prior_count=1,
                                    robust=True))]),
                         col_trf_col_grps[0][0])])),
                ('trf1', StandardScaler()),
                ('clf2', LogisticRegression(
                    class_weight='balanced',
                    max_iter=5000,
                    penalty='l2',
                    random_state=random_seed,
                    solver='saga'))])
        param_grid_dict = {'clf2__C': clf_c,
                           'trf0__trf0__slr1__k': skb_k}
    
    param_grid = [param_grid_dict.copy()]
    return pipe, param_grid, param_grid_dict
