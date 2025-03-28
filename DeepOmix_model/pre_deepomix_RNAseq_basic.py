import mygene

mg = mygene.MyGeneInfo()
gene_symbols = ['TP53', 'EGFR', 'BRCA1']
result = mg.querymany(gene_symbols, scopes='symbol', fields='entrezgene', species='human')
# Output: Map symbols to Entrez IDs (e.g., TP53 → 7157)

import gseapy as gp
import pandas as pd

# Fetch KEGG pathways for human
pathway_db = 'KEGG_2021_Human'
pathways = gp.get_library(name=pathway_db)

# Create a gene-pathway matrix
genes = [...]  # List of your gene IDs (e.g., Entrez)
pathway_mask = pd.DataFrame(0, index=genes, columns=pathways.keys())

for pathway, gene_list in pathways.items():
    pathway_mask.loc[pathway_mask.index.isin(gene_list), pathway] = 1

# Save as a CSV or use directly in DeepOmix
pathway_mask.to_csv('gene_pathway_matrix.csv')

coverage = pathway_mask.sum(axis=1)  # Number of pathways each gene belongs to
print(f"Genes not in any pathway: {len(coverage[coverage == 0])}")
print(f"Median pathways per gene: {coverage.median()}")

import torch

# Load the gene-pathway matrix
pathway_mask_df = pd.read_csv('gene_pathway_matrix.csv', index_col=0)
pathway_mask_tensor = torch.tensor(pathway_mask_df.values, dtype=torch.float32)

# Initialize DeepOmix with the mask
model = DeepOmix(
    n_genes=pathway_mask_df.shape[0],
    n_pathways=pathway_mask_df.shape[1],
    pathway_mask=pathway_mask_tensor
)




