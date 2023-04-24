#include "compute_residual.h"
#include "restrict.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    int N = 1001;
    int n = (N - 1) / 2 + 1;
    int curn = n;
    int sn = (n - 1) / 2 + 1;
    double ***phi = new double **[2];
    double ***aux = new double **[2];
    double ***f = new double **[2];
    for (int i = 0; i < 2; i++)
    {
        phi[i] = new double *[curn];
        aux[i] = new double *[curn];
        f[i] = new double *[curn];
        for (int r = 0; r < curn; r++)
        {
            phi[i][r] = new double[curn];
            aux[i][r] = new double[curn];
            f[i][r] = new double[curn];
        }
        curn = (curn - 1) / 2 + 1;
    }
    // initialize the corresponding value
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            phi[0][i][j] = 0;
            aux[0][i][j] = 0;
            f[0][i][j] = 1;
        }
    }

    // apply the restrict function
    restrict(phi, f, aux, n, 0, comm);
    printf("rank%d restrict completed\n", rank);
    MPI_Barrier(comm);
    double boundaryv1, boundaryv2, centerv;
    if (rank == 0)
    {
        boundaryv1 = f[1][sn - 1][2];
        boundaryv2 = f[1][2][sn - 1];
        centerv = f[1][sn - 1][sn - 1];
    }
    else if (rank == 1)
    {
        boundaryv1 = f[1][sn - 1][2];
        boundaryv2 = f[1][2][0];
        centerv = f[1][sn - 1][0];
    }
    else if (rank == 2)
    {
        boundaryv1 = f[1][0][2];
        boundaryv2 = f[1][2][0];
        centerv = f[1][0][0];
    }
    else
    {
        boundaryv1 = f[1][0][2];
        boundaryv2 = f[1][2][sn - 1];
        centerv = f[1][0][sn - 1];
    }
    printf("in rank %d,bdv1 = %f,bdv2 = %f,centerv = %f\n", rank, boundaryv1, boundaryv2, centerv);
    int nds = n;
    for (int ilvl = 0; ilvl < 2; ilvl++)
    {
        for (int i = 0; i < nds; i++)
        {
            delete[] phi[ilvl][i];
            delete[] aux[ilvl][i];
            delete[] f[ilvl][i];
        }
        delete[] phi[ilvl];
        delete[] aux[ilvl];
        delete[] f[ilvl];
        nds = (nds - 1) / 2 + 1;
    }
    delete[] phi;
    delete[] aux;
    delete[] f;
    MPI_Finalize();
}