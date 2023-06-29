import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

start_time = MPI.Wtime()
MPI.Init()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sum = 0
cal_number = int(1000/size)
start_value = rank * cal_number
for i in range(1, cal_number+1):
    sum = sum + start_value + i
total_sum = comm.gather(sum, root=0)
if rank==0:
    total = 0
    for num in total_sum:
        total = total + num
    print("Sum: {}".format(total))
MPI.Finalize()
end_time = MPI.Wtime()
print("Id:{}, Time useage:{}".format(rank, end_time - start_time))

