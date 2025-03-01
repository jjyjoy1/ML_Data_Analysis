def handle_model_type(args, analysis):
    """
    Validate and adjust model type based on analysis type to ensure compatibility.
    
    Parameters:
    -----------
    args : argparse.Namespace
        The command-line arguments
    analysis : str
        The analysis type ('surv' or 'resp')
        
    Returns:
    --------
    str
        The validated/adjusted model type
    """
    # Define valid model types for each analysis
    survival_models = ['cnet', 'rsf', 'svm']
    response_models = ['rfe', 'lgr', 'edger', 'limma', 'rf', 'gb']
    
    # Handle survival analysis
    if analysis == 'surv':
        if args.model_type not in survival_models:
            print(f"Warning: Model type '{args.model_type}' is not compatible with survival analysis.")
            print(f"Using default survival model: 'cnet'")
            return 'cnet'
        return args.model_type
    
    # Handle drug response analysis
    elif analysis == 'resp':
        if args.model_type not in response_models:
            print(f"Warning: Model type '{args.model_type}' is not compatible with drug response analysis.")
            print(f"Using default response model: 'lgr'")
            return 'lgr'
        return args.model_type
    
    # For any other analysis type
    return args.model_type

# Example usage in main script:
# args.model_type = handle_model_type(args, analysis)
