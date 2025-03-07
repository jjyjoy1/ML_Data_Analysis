"""
Convert R ExpressionSet objects to Python AnnData objects.

This script provides detailed handling for converting TCGA ExpressionSets 
created by the create_esets.R script into equivalent AnnData objects,
preserving all metadata and annotations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import logging
import warnings

# rpy2 imports for interfacing with R
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri, conversion
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
except ImportError:
    sys.exit("Error: rpy2 package is required. Please install it with: pip install rpy2")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eset2anndata')

# Silence R warnings
rpy2_logger.setLevel(logging.ERROR)

# Activate automatic conversion
pandas2ri.activate()
numpy2ri.activate()

def setup_r_environment():
    """
    Set up the R environment and import necessary packages.
    
    Returns:
        tuple: Imported R packages (base, Biobase, gdata)
    """
    logger.info("Setting up R environment")
    
    # Check for required R packages
    required_packages = ['Biobase', 'gdata']
    
    try:
        # Import R's base package
        base = importr('base')
        utils = importr('utils')
        
        # Check if required packages are installed
        r_check_installed = ro.r('function(pkg) is.element(pkg, installed.packages()[,1])')
        
        for package in required_packages:
            if not r_check_installed(package)[0]:
                logger.error(f"R package '{package}' is not installed. Please install it in R with: install.packages('{package}')")
                sys.exit(1)
        
        # Import required R packages
        biobase = importr('Biobase')
        gdata = importr('gdata')
        
        logger.info("R environment ready")
        return base, biobase, gdata
    
    except Exception as e:
        logger.error(f"Failed to set up R environment: {e}")
        sys.exit(1)

def convert_eset_to_anndata(eset_file, output_dir, r_packages=None):
    """
    Convert an R ExpressionSet object to an AnnData object with full metadata preservation.
    
    Args:
        eset_file (str): Path to RDS file containing an ExpressionSet
        output_dir (str): Directory to save the AnnData object
        r_packages (tuple, optional): Tuple of imported R packages. If None, will be imported.
    
    Returns:
        str: Path to saved AnnData file, or None if conversion failed
    """
    if not os.path.exists(eset_file):
        logger.error(f"File not found: {eset_file}")
        return None
    
    # Import R packages if not provided
    if r_packages is None:
        base, biobase, gdata = setup_r_environment()
    else:
        base, biobase, gdata = r_packages
    
    logger.info(f"Converting {os.path.basename(eset_file)}")
    
    try:
        # Load the ExpressionSet
        eset = base.readRDS(eset_file)
        
        # Verify it's an ExpressionSet
        is_eset = ro.r("is")(eset, "ExpressionSet")[0]
        if not is_eset:
            logger.warning(f"{eset_file} is not an ExpressionSet, skipping")
            return None
        
        # Get expression matrix (features as rows, samples as columns)
        exprs_matrix = biobase.exprs(eset)
        X = np.array(exprs_matrix).T  # Transpose for AnnData (samples as rows)
        
        # Get phenotype data (sample metadata)
        with localconverter(ro.default_converter + pandas2ri.converter):
            pdata = biobase.pData(eset)
            obs = conversion.rpy2py(pdata)
        
        # Get feature data if available
        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                fdata = biobase.fData(eset)
                var = conversion.rpy2py(fdata)
        except Exception as e:
            logger.info(f"No feature data found, creating basic feature info: {e}")
            # Create basic feature info
            feature_names = list(biobase.featureNames(eset))
            var = pd.DataFrame(index=feature_names)
        
        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Add additional metadata
        # Extract experimental metadata
        try:
            adata.uns['experimentData'] = {
                'name': str(ro.r('experimentData(eset)@name')[0]) if ro.r('exists("experimentData(eset)@name")')[0] else "",
                'lab': str(ro.r('experimentData(eset)@lab')[0]) if ro.r('exists("experimentData(eset)@lab")')[0] else "",
                'contact': str(ro.r('experimentData(eset)@contact')[0]) if ro.r('exists("experimentData(eset)@contact")')[0] else "",
                'title': str(ro.r('experimentData(eset)@title')[0]) if ro.r('exists("experimentData(eset)@title")')[0] else "",
                'abstract': str(ro.r('experimentData(eset)@abstract')[0]) if ro.r('exists("experimentData(eset)@abstract")')[0] else ""
            }
        except Exception as e:
            logger.debug(f"Could not extract experiment metadata: {e}")
        
        # Store dataset type information
        filename = os.path.basename(eset_file)
        if "_surv_" in filename:
            adata.uns['dataset_type'] = 'survival'
            # Make sure survival-specific columns are properly formatted
            for col in ['Status', 'Survival_in_days']:
                if col in adata.obs.columns:
                    if col == 'Status':
                        adata.obs[col] = adata.obs[col].astype('category')
                    else:
                        adata.obs[col] = pd.to_numeric(adata.obs[col])
        
        elif "_resp_" in filename:
            adata.uns['dataset_type'] = 'response'
            # Make sure Class column is properly formatted
            if 'Class' in adata.obs.columns:
                adata.obs['Class'] = adata.obs['Class'].astype('category')
        
        # Store data source information
        if "_kraken_" in filename:
            adata.uns['data_source'] = 'kraken'
        elif "_combo_" in filename:
            adata.uns['data_source'] = 'combo'
        elif "HTSeq" in filename:
            adata.uns['data_source'] = 'htseq'
        
        # Extract cancer type from filename
        cancer_match = ro.r('regmatches')(filename, ro.r('regexpr')('tcga-[a-z]+', filename.lower()))
        if len(cancer_match) > 0:
            adata.uns['cancer_type'] = str(cancer_match[0]).upper()
        
        # Generate output path
        output_filename = os.path.splitext(os.path.basename(eset_file))[0].replace('_eset', '_anndata') + '.h5ad'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save AnnData object
        adata.write(output_path)
        
        logger.info(f"Successfully saved {output_path}")
        logger.info(f"  - Observations (samples): {adata.n_obs}")
        logger.info(f"  - Variables (features): {adata.n_vars}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting {eset_file}: {e}")
        return None

def find_eset_files(input_dir, pattern="*_eset.rds"):
    """
    Find all ExpressionSet files matching the pattern.
    
    Args:
        input_dir (str): Directory to search
        pattern (str): Glob pattern to match files
    
    Returns:
        list: List of file paths
    """
    return sorted(list(Path(input_dir).glob(pattern)))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert R ExpressionSet objects to AnnData")
    parser.add_argument("--input-dir", default="data", help="Directory containing R ExpressionSet objects")
    parser.add_argument("--output-dir", default="data", help="Directory to save AnnData objects")
    parser.add_argument("--pattern", default="*_eset.rds", help="Filename pattern to match ExpressionSet files")
    parser.add_argument("--cancer-types", nargs="+", help="Specific cancer types to convert (e.g., TCGA-BRCA)")
    parser.add_argument("--dataset-types", nargs="+", choices=["surv", "resp"], help="Dataset types to convert")
    parser.add_argument("--data-sources", nargs="+", choices=["kraken", "htseq", "combo"], help="Data sources to convert")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def main():
    """Main function to convert ExpressionSet files to AnnData."""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up R environment
    r_packages = setup_r_environment()
    
    # Find ExpressionSet files
    all_eset_files = find_eset_files(args.input_dir, args.pattern)
    logger.info(f"Found {len(all_eset_files)} ExpressionSet files in {args.input_dir}")
    
    # Filter files based on command-line arguments
    eset_files = all_eset_files.copy()
    
    if args.cancer_types:
        cancer_patterns = [ct.lower() for ct in args.cancer_types]
        eset_files = [f for f in eset_files if any(cp in str(f).lower() for cp in cancer_patterns)]
        logger.info(f"Filtered to {len(eset_files)} files matching cancer types: {args.cancer_types}")
    
    if args.dataset_types:
        dataset_patterns = [f"_{dt}_" for dt in args.dataset_types]
        eset_files = [f for f in eset_files if any(dp in str(f).lower() for dp in dataset_patterns)]
        logger.info(f"Filtered to {len(eset_files)} files matching dataset types: {args.dataset_types}")
    
    if args.data_sources:
        source_patterns = [f"_{ds}_" for ds in args.data_sources]
        eset_files = [f for f in eset_files if any(sp in str(f).lower() for sp in source_patterns)]
        logger.info(f"Filtered to {len(eset_files)} files matching data sources: {args.data_sources}")
    
    # Convert each ExpressionSet file
    converted_files = []
    for eset_file in eset_files:
        result = convert_eset_to_anndata(eset_file, args.output_dir, r_packages)
        if result:
            converted_files.append(result)
    
    logger.info(f"Successfully converted {len(converted_files)} of {len(eset_files)} files")
    
    # Print summary of conversions
    if converted_files:
        logger.info("Conversion summary:")
        for file in converted_files:
            logger.info(f"  - {os.path.basename(file)}")
    
    return 0

if __name__ == "__main__":
    main()


