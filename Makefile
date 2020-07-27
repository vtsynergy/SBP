INCLUDES=-I./argparse/include -I./include

CC=g++ -std=c++17

.PHONY: init_parser

# argparse is a 3rd-party library for parsing command-line arguments. Added as a submodule so users don't
# have to download it, so it needs to be initialized
init_parser:
	cd argparse
	git submodule init
	git submodule update

main: init_parser main.cpp
	$(CC) -o main main.cpp graph.cpp utils.cpp $(INCLUDES)

runmain: main
	./main -d ../graph_challenge/data
	rm main
