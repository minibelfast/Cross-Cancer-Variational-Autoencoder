library(monocle, lib.loc='/data1/Rpackages')
library(Matrix)

print("Loading data...")
expr_matrix <- readMM('monocle_export/counts_matrix.mtx')
# 关键修复1：确保它是压缩的列稀疏矩阵格式，Monocle2能处理dgCMatrix
expr_matrix <- as(expr_matrix, "dgCMatrix")

metadata <- read.csv('monocle_export/metadata.csv', row.names=1)
genes <- read.csv('monocle_export/genes.csv', header=FALSE)
rownames(expr_matrix) <- genes$V1
colnames(expr_matrix) <- rownames(metadata)

print(paste("Matrix dimensions:", nrow(expr_matrix), "genes x", ncol(expr_matrix), "cells"))

print("Creating CellDataSet...")
fd <- data.frame(gene_short_name = rownames(expr_matrix), row.names = rownames(expr_matrix))
pd <- new('AnnotatedDataFrame', data = metadata)
fd <- new('AnnotatedDataFrame', data = fd)

# 关键修复2：千万不要加 as.matrix()！直接传稀疏矩阵，否则会把0全部填满导致内存爆炸
cds <- newCellDataSet(expr_matrix, phenoData = pd, featureData = fd, expressionFamily=negbinomial.size())

print("Estimating size factors...")
cds <- estimateSizeFactors(cds)

print("Estimating dispersions...")
cds <- estimateDispersions(cds)

print("Selecting genes for ordering...")
disp_table <- dispersionTable(cds)
# 稍微过滤一下基因，加快后续降维速度
ordering_genes <- subset(disp_table, mean_expression >= 0.1 & dispersion_empirical >= 1 * dispersion_fit)$gene_id
cds <- setOrderingFilter(cds, ordering_genes)

print("Reducing dimension (DDRTree)...")
# 为了防止 DDRTree 内部转矩阵导致内存溢出，使用 FIt-SNE 或者限制 max_components
cds <- reduceDimension(cds, max_components = 2, method = 'DDRTree')

print("Ordering cells...")
cds <- orderCells(cds)

print("Plotting trajectory...")
pdf("monocle_export/Trajectory_Plot.pdf", width=8, height=6)
plot_cell_trajectory(cds, color_by = "Cancer_Type")
plot_cell_trajectory(cds, color_by = "leiden_ccvae")
dev.off()

print("Monocle2 analysis completed successfully!")
