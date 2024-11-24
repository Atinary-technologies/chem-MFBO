#!/bin/bash

source venv/bin/activate

echo "Running cofs..."
python src/chem_mfbo/benchmark/cofs.py
if [ $? -ne 0 ]; then
    echo "Error running cofs.py. Exiting."
    exit 1
fi

echo "Running polarizability..."
python src/chem_mfbo/benchmark/polarizability.py
if [ $? -ne 0 ]; then
    echo "Error running polarizability.py. Exiting."
    exit 1
fi


echo "Running freesolv..."
python src/chem_mfbo/benchmark/freesolv.py
if [ $? -ne 0 ]; then
    echo "Error running freesolv.py. Exiting."
    exit 1
fi


echo "Running polarizability bad..."
python src/chem_mfbo/benchmark/polarizability.py --config-name=polarizability_bad
if [ $? -ne 0 ]; then
    echo "Error running freesolv.py. Exiting."
    exit 1
fi


echo "All scripts completed successfully!"
