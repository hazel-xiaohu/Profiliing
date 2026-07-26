numerical_file=PanguLU
Smatrix_name=$1
NP=$2
NB=${3:-32}
if [ ! -f $1 ];then
  echo "the $1 is not a file"
  exit
fi

export HWLOC_COMPONENTS=${HWLOC_COMPONENTS:--gl}

echo mpirun -np $[$NP] ../test/$numerical_file -F $Smatrix_name -O $NB

mpirun -np $[$NP] ../test/$numerical_file -F $Smatrix_name -O $NB

#warning  

#the host file save the node use in Distributed Systems

#echo mpirun -np $[$NP] -hostfile host ../test/$numerical_file -F $Smatrix_name

#mpirun -np $[$NP] -hostfile host ../test/$numerical_file -F $Smatrix_name
