import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize =False
import numpy as np
from mpi4py import MPI

start_time = MPI.Wtime()
N = 10000
MPI.Init()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
m, n = N, int(N/size)

# allocate memeory
local_C = np.zeros(shape=(n, m))

# data init
x_flat = np.linspace(0, 1, m, endpoint=True)
y_flat = x_flat[rank * n: (rank + 1) * n]
X, Y = np.meshgrid(x_flat, y_flat)
local_A = np.exp(Y) * np.sin(3 * X)
X, Y = np.meshgrid(y_flat, x_flat)
local_B = (X + np.cos(4 * X))*(1 + Y)
local_C[:, rank * n : (rank + 1) * n] = np.matmul(local_A, local_B)


for i in range(1, size):
    send_id = (rank + i + size) % size
    recv_id = (rank - i + size) % size
    recvbuf = np.zeros_like(local_B)
    comm.Sendrecv(sendbuf=local_B, dest=send_id, recvbuf=recvbuf, source=recv_id)
    local_C[:, recv_id * n : (recv_id + 1) * n] = np.matmul(local_A, recvbuf)
local_sum = np.square(local_C).sum()
global_sum = comm.gather(local_sum, 0)

if rank == 0:
    avg = np.array(global_sum).sum()/N/N
    print("S = {}".format(avg))
MPI.Finalize()
end_time = MPI.Wtime()
print("Id:{}, Time useage:{}".format(rank, end_time - start_time))
