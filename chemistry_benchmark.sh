#!/bin/bash

source venv/bin/activate

echo "Running cofs..."
python src/mf_kmc/benchmark/cofs.py
if [ $? -ne 0 ]; then
    echo "Error running cofs.py. Exiting."
    exit 1
fi

# Run the second script
echo "Running polarizability..."
python src/mf_kmc/benchmark/polarizability.py
if [ $? -ne 0 ]; then
    echo "Error running polarizability.py. Exiting."
    exit 1
fi

# Run the third script
echo "Running freesolv..."
python src/mf_kmc/benchmark/freesolv.py
if [ $? -ne 0 ]; then
    echo "Error running freesolv.py. Exiting."
    exit 1
fi

# Run the third script
echo "Running kinetic..."
python src/mf_kmc/benchmark/kinetic.py
if [ $? -ne 0 ]; then
    echo "Error running kinetic.py. Exiting."
    exit 1
fi

echo "All scripts completed successfully!"
