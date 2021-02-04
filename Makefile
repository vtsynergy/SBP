PATCHES=-I./patches
TCLAP=-I./tclap-1.2.2/include
INCLUDES=-I./include -I./include/sbp -I./include/sbp/blockmodel -I./include/sbp/blockmodel/sparse -I./hungarian $(PATCHES) $(TCLAP)
CC=g++ -std=c++17 -O3 -no-pie
MPICC=mpic++ -std=c++17 -O3 -no-pie
DEBUG=-g3 -fsanitize=address
BLOCKMODELDIR=sbp/blockmodel
SPARSEDIR=$(BLOCKMODELDIR)/sparse
BLOCKMODEL_FILES=$(BLOCKMODELDIR)/blockmodel.cpp $(BLOCKMODELDIR)/blockmodel_triplet.cpp
SBP_FILES=sbp/sbp.cpp sbp/block_merge.cpp sbp/common.cpp sbp/finetune.cpp sbp/mpi_utils.cpp
SPARSE_FILES=$(SPARSEDIR)/boost_mapped_matrix.cpp $(SPARSEDIR)/dict_matrix.cpp $(SPARSEDIR)/dict_transpose_matrix.cpp
FILES=main.cpp graph.cpp utils.cpp $(SBP_FILES) partition.cpp $(BLOCKMODEL_FILES) $(SPARSE_FILES) evaluate.cpp
FILES+=hung.o

hung.o:
	$(CC) -o hung.o -c hungarian/hungarian.cpp $(PATCHES)

main: hung.o main.cpp
	$(MPICC) -fopenmp -o main $(FILES) $(INCLUDES) -lstdc++fs

debugmain: hung.o main.cpp
	$(MPICC) $(DEBUG) -fopenmp -o main $(FILES) $(INCLUDES) -lstdc++fs

runmain: main
	./main -d ../graph_challenge/data
	rm main hung.o
