#!/bin/bash

source venv/bin/activate

echo "Running cofs..."
python src/chem_mfbo/benchmark/real_problems.py --config-name=cofs
if [ $? -ne 0 ]; then
    echo "Error running cofs. Exiting."
    exit 1
fi

echo "Running polarizability..."
python src/chem_mfbo/benchmark/real_problems.py --config-name=polarizability
if [ $? -ne 0 ]; then
    echo "Error running polarizability. Exiting."
    exit 1
fi


echo "Running freesolv..."
python src/chem_mfbo/benchmark/real_problems.py --config-name=freesolv
if [ $? -ne 0 ]; then
    echo "Error running freesolv.py. Exiting."
    exit 1
fi

echo "Running polarizability bad..."
python src/chem_mfbo/benchmark/real_problems.py --config-name=polarizability_bad
if [ $? -ne 0 ]; then
    echo "Error running polarizability bad. Exiting."
    exit 1
fi


echo "All scripts completed successfully!"
