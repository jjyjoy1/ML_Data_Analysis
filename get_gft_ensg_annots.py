import pandas as pd
import gffutils
###pip install pandas gffutils
#Extract gtf file from local file

#Tracking gtf file from GDC
#https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files


def extract_gene_annotations(gtf_file, output_file):
    # Create a database from the GTF file
    db = gffutils.create_db(gtf_file, dbfn=':memory:', force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)

    # Initialize lists to store gene information
    gene_ids = []
    gene_names = []
    gene_types = []

    # Iterate through genes in the database
    for gene in db.features_of_type('gene'):
        gene_id = gene.id
        gene_name = gene.attributes.get('gene_name', [''])[0]
        gene_type = gene.attributes.get('gene_type', [''])[0]

        gene_ids.append(gene_id)
        gene_names.append(gene_name)
        gene_types.append(gene_type)

    # Create a DataFrame with the gene information
    gene_df = pd.DataFrame({
        'gene_id': gene_ids,
        'gene_name': gene_names,
        'gene_type': gene_types
    })
    # Save the DataFrame to a CSV file
    gene_df.to_csv(output_file, index=False, set= '\t')



import os
import requests

def download_gtf(url, local_path):
    """Downloads a GTF file from a URL if not already present."""
    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"File saved to {local_path}")
        else:
            raise Exception(f"Failed to download GTF file: {response.status_code}")


gtf_file = 'https://api.gdc.cancer.gov/data/25aa497c-e615-4cb7-8751-71f744f9691f'
gtf_local_path = '/home/jjy/Downloads/Tcga_Mega/tcga-microbiome-prediction/CustomCode/input.gtf.gz'

download_gtf(gtf_url, gtf_local_path)

import os
import gffutils

def extract_gene_annotations(gtf_file, output_file):
    # Check if the file exists before proceeding
    if not os.path.exists(gtf_file):
        raise FileNotFoundError(f"Error: The file '{gtf_file}' was not found.")

    # Create a database from the GTF file
    db = gffutils.create_db(gtf_file, dbfn=':memory:', force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)

    # Process the gene annotations
    gene_ids, gene_names, gene_types = [], [], []
    for gene in db.features_of_type('gene'):
        gene_id = gene.id
        gene_name = gene.attributes.get('gene_name', [''])[0]
        gene_type = gene.attributes.get('gene_type', [''])[0]

        gene_ids.append(gene_id)
        gene_names.append(gene_name)
        gene_types.append(gene_type)

    # Save to a file
    with open(output_file, 'w') as f:
        f.write("gene_id,gene_name,gene_type\n")
        for i in range(len(gene_ids)):
            f.write(f"{gene_ids[i]},{gene_names[i]},{gene_types[i]}\n")

# Example usage
output_file = 'processed_gene_annotations.csv'

extract_gene_annotations(gtf_file, output_file)



