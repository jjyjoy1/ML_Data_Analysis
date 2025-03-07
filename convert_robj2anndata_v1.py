#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
import glob
from pathlib import Path

# Enable automatic conversion between R and Python objects
pandas2ri.activate()
numpy2ri.activate()

# Import required R packages
base = importr('base')
biobase = importr('Biobase')
utils = importr('utils')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert R ExpressionSet objects to AnnData")
    parser.add_argument("--input-dir", default="data", help="Directory containing R ExpressionSet objects")
    parser.add_argument("--output-dir", default="data", help="Directory to save AnnData objects")
    parser.add_argument("--pattern", default="*_eset.rds", help="Filename pattern to match ExpressionSet files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def convert_eset_to_anndata(eset_file, output_dir, verbose=False):
    """
    Convert an R ExpressionSet object to an AnnData object.
    
    Parameters
    ----------
    eset_file : str
        Path to the RDS file containing an ExpressionSet object
    output_dir : str
        Directory to save the AnnData object
    verbose : bool
        Whether to print additional information
        
    Returns
    -------
    str
        Path to the saved AnnData file
    """
    if verbose:
        print(f"Converting {eset_file}")
    
    # Load the ExpressionSet from RDS
    eset = base.readRDS(eset_file)
    
    # Check if the object is an ExpressionSet
    try:
        is_eset = ro.r("is")(eset, "ExpressionSet")[0]
        if not is_eset:
            print(f"Warning: {eset_file} is not an ExpressionSet, skipping")
            return None
    except Exception as e:
        print(f"Error checking if {eset_file} is an ExpressionSet: {e}")
        return None
    
    try:
        # Extract expression matrix
        exprs = biobase.exprs(eset)
        # Expression matrix has features (genes) as rows and samples as columns in the ExpressionSet
        # AnnData expects samples as rows and features as columns, so we need to transpose
        X = np.array(exprs).T
        
        # Extract phenotype data
        pdata = biobase.pData(eset)
        obs = pd.DataFrame(pdata)
        
        # Extract feature data if available
        try:
            fdata = biobase.fData(eset)
            var = pd.DataFrame(fdata)
        except:
            # If fData is not available, create a simple DataFrame with feature names
            var = pd.DataFrame(index=pd.Index(biobase.featureNames(eset), name='gene_id'))
        
        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Add any additional metadata from the ExpressionSet
        try:
            # Get experiment metadata
            experiment_data = biobase.experimentData(eset)
            if experiment_data is not None:
                # Extract metadata fields 
                adata.uns['experimentData'] = {
                    'name': str(ro.r("experimentData(eset)@name")[0]) if ro.r("exists('experimentData(eset)@name')")[0] else "",
                    'lab': str(ro.r("experimentData(eset)@lab")[0]) if ro.r("exists('experimentData(eset)@lab')")[0] else "",
                    'contact': str(ro.r("experimentData(eset)@contact")[0]) if ro.r("exists('experimentData(eset)@contact')")[0] else "",
                    'title': str(ro.r("experimentData(eset)@title")[0]) if ro.r("exists('experimentData(eset)@title')")[0] else "",
                    'abstract': str(ro.r("experimentData(eset)@abstract")[0]) if ro.r("exists('experimentData(eset)@abstract')")[0] else ""
                }
        except Exception as e:
            if verbose:
                print(f"Warning: Could not extract experiment metadata: {e}")
        
        # Get protocol data if available
        try:
            protocol_data = biobase.protocolData(eset)
            if protocol_data is not None:
                protocol_df = pd.DataFrame(protocol_data)
                for col in protocol_df.columns:
                    adata.obs[f"protocol_{col}"] = protocol_df[col].values
        except Exception as e:
            if verbose:
                print(f"Warning: Could not extract protocol data: {e}")
                
        # Generate output filename
        eset_basename = os.path.basename(eset_file)
        anndata_filename = os.path.splitext(eset_basename)[0].replace('_eset', '_anndata') + '.h5ad'
        anndata_path = os.path.join(output_dir, anndata_filename)
        
        # Save the AnnData object
        adata.write(anndata_path)
        
        if verbose:
            print(f"Saved {anndata_path}")
            print(f"  - Samples: {adata.n_obs}")
            print(f"  - Features: {adata.n_vars}")
        
        return anndata_path
    
    except Exception as e:
        print(f"Error converting {eset_file}: {e}")
        return None

def main():
    """Main function to convert all ExpressionSet files to AnnData."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all ExpressionSet files matching the pattern
    eset_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    
    if args.verbose:
        print(f"Found {len(eset_files)} ExpressionSet files to convert")
    
    # Convert each ExpressionSet file to AnnData
    converted_files = []
    for eset_file in eset_files:
        result = convert_eset_to_anndata(eset_file, args.output_dir, args.verbose)
        if result:
            converted_files.append(result)
    
    print(f"Successfully converted {len(converted_files)} of {len(eset_files)} files")
    
    return 0

if __name__ == "__main__":
    main()


