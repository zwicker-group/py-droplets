#!/bin/bash
#
# Requires the `nose-timer` package to time the tests 
#

CORES=`python3 -c 'from multiprocessing import cpu_count; print(cpu_count() // 2)'`

export PYTHONPATH=../../py-pde:$PYTHONPATH
export MPLBACKEND="agg"

echo 'Run unittests on '$CORES' cores:'
cd ..
nosetests --processes=$CORES --process-timeout=60 --stop \
	--with-timer --timer-top-n 10 --timer-ok 500ms --timer-warning 1
