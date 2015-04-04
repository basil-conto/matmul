# Complex matrix multiplication using Open MPI
Basil L. Contovounesios <contovob@tcd.ie>

##### Multiplication optimisation

All multiplication in the matmul program is row-oriented and strives for more
efficient execution using a couple of techniques. The first is to make local
copies of every column in the second matrix, B, before operating on them. This
effectively transposes the second matrix with respect to its layout in memory,
thus taking advantage of spatial locality. The second technique aims to mitigate
the latency of memory access by storing each component of the matrix involved in
the multiplication in a local variable before operating on it.

##### Communication model

The program assumes a master/slave communication model where the master node
essentially runs the harness program and the slaves await work to be sent their
way. Thus only the master generates the matrices and performs error handling on
the program input. There is one breach of this model - all nodes parse their
matrix size arguments. This avoids any communication overhead associated with
small enough input which should not be parallelised.

##### Parallelism approach

The program defines a constant matrix size threshold below which the
multiplication is carried out by the master node alone. When this threshold is
exceeded, the work must be distributed amongst the slave nodes as evenly as
possible. Due to the row-oriented, rather than block-oriented, nature of the
program, the simplest way of dividing the matrices is to divide the larger of
the two dimensions (rows in A) and (columns in B) by the number of nodes.

This approach becomes slightly problematic when neither of the two dimensions is
a multiple of the number of nodes. The solution in this case is to have two
steppings in the division. For example, 6 rows can be shared by 4 nodes in a
{2, 2, 1, 1} configuration.

Thus the matrix with the smaller dimension is broadcast to all nodes in its
entirety, whereas the matrix with the larger dimension is sent in slivers to
each slave node individually, based on their rank. The master node, in turn,
operates on one of the smaller slivers itself before gathering the results from
the slave nodes.
