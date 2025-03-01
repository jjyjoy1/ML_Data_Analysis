###This code is useful to review R ExperimentSet file "rds" contains; and also enable to convert one rds file to three table file which are pdata, assaydata abd fdata from ExperimentSet; and also enable to convert rds file to h5ad file 

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

pandas2ri.activate()  # Enable Pandas conversion

def extract_expressionset_data(rds_path):
    """

    2. Sample metadata (samples x metadata fields)

        rds_path (str): Path to the RDS file
    Returns:
        exprs_df (pd.DataFrame): Expression matrix
        pheno_df (pd.DataFrame): Sample metadata
        feature_df (pd.DataFrame): Feature metadata
    """
    # Load the RDS file
    r_object = robjects.r['readRDS'](rds_path)
    # Extract expression matrix
    exprs_matrix = robjects.r["exprs"](r_object)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        exprs_df = pd.DataFrame(robjects.conversion.rpy2py(exprs_matrix))
    # Extract sample metadata (phenoData)
    pheno_data = robjects.r["pData"](r_object)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pheno_df = pd.DataFrame(robjects.conversion.rpy2py(pheno_data))
    # Extract feature metadata (featureData)
    feature_data = robjects.r["fData"](r_object)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        feature_df = pd.DataFrame(robjects.conversion.rpy2py(feature_data))
    # Get feature names (rows) for exprs_df
    feature_names = robjects.r["featureNames"](r_object)
    if feature_names is not None and feature_names.size > 0:
        exprs_df.index = feature_names
    else:
        exprs_df.index = [f"Feature_{i}" for i in range(exprs_df.shape[0])]
    # Get sample names (columns) for exprs_df and pheno_df
    sample_names = robjects.r["sampleNames"](r_object)
    if sample_names is not None and feature_names.size > 0:
        exprs_df.columns = sample_names
        pheno_df.index = sample_names
    else:
        exprs_df.columns = [f"Sample_{i}" for i in range(exprs_df.shape[1])]
        pheno_df.index = exprs_df.columns  # Use the same names
    # Get feature names (rows) for feature_df
    feature_df.index = exprs_df.index
    return exprs_df, pheno_df, feature_df


exprs_df, pheno_df, feature_df = extract_expressionset_data(rds_file)

# Print shapes to confirm
print("Expression Matrix Shape:", exprs_df.shape)
print("Sample Metadata Shape:", pheno_df.shape)
print("Feature Metadata Shape:", feature_df.shape)

# Preview tables
print(exprs_df.head())    # Expression data (gene/microbe abundance)
print(pheno_df.head())    # Sample metadata (clinical variables)
print(feature_df.head())  # Feature annotations


###convert three tables to r object.
pandas2ri.activate()

def convert_to_r_expressionset(exprs_df, pheno_df, feature_df):
    """
    Converts three Pandas DataFrames (expression matrix, sample metadata, feature metadata)
    into an R ExpressionSet object.
    
    Parameters:
        exprs_df (pd.DataFrame): Expression matrix (features x samples)
        pheno_df (pd.DataFrame): Sample metadata (samples x metadata fields)
        feature_df (pd.DataFrame): Feature metadata (features x annotations)
    
    Returns:
        R ExpressionSet object
    """
    with localconverter(robjects.default_converter + pandas2ri.converter):
        # Convert expression matrix to R matrix
        exprs_r = robjects.r.matrix(
            robjects.FloatVector(exprs_df.values.flatten()), 
            nrow=exprs_df.shape[0], 
            dimnames=[robjects.StrVector(exprs_df.index), robjects.StrVector(exprs_df.columns)]
        )
        
        # Convert sample metadata to R DataFrame
        pheno_r = pandas2ri.py2rpy(pheno_df)
        
        # Convert feature metadata to R DataFrame
        feature_r = pandas2ri.py2rpy(feature_df)

    # Load the Bioconductor package 'Biobase' for ExpressionSet
    robjects.r('library(Biobase)')

    # Create the ExpressionSet in R
    exprs_set_r = robjects.r['ExpressionSet'](
        assayData=robjects.r['list'](exprs=exprs_r),
        phenoData=robjects.r['AnnotatedDataFrame'](pheno_r),
        featureData=robjects.r['AnnotatedDataFrame'](feature_r)
    )
    
    return exprs_set_r


r_expressionset = convert_to_r_expressionset(exprs_df, pheno_df, feature_df)

# Check if the ExpressionSet was successfully created
print(r_expressionset)

output_rds_path = "modified_expressionset.rds"

# Save the R object as an RDS file
robjects.r['saveRDS'](r_expressionset, output_rds_path)

print(f"Modified ExpressionSet saved to {output_rds_path}")

################################################################




