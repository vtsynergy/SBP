# Sets the default rule to install the cppsbp library
.DEFAULT_GOAL := default
default:
	cd utils/cpp; \
	python setup.py install; \
	cd ../..;
	make clean;

# Runs stochastic block partitioning on the 1K vertex graph with no sampling
test:
	python partition_baseline_main.py;

# Deletes locally generated files
clean:
	cd utils/cpp; \
	rm -rf build; \
	rm -rf cppsbp.egg-info; \
	rm -rf dist; \
	rm -rf tmp;

# Rebuilds and then tests everything
rerun: default
	make test;
