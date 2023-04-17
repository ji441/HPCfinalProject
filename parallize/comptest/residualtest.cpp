#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <cmath>
#include <iostream>
#include "../compute_residual.h"
using namespace std;
//mpic++ -std=c++11 -O3 -march=native -fopenmp residualtest.cpp -o testres
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_WORLD;
    int p;//number of nodes/processes we have.
    MPI_Comm_size(comm, &p);
    if (p != 4) {
        printf("number of processes must be 4");
        abort();
    }
    int N = 1001;
    int n = (N - 1) / 2 + 1;
    //allocate arrays
    double** phi = new double* [n];
    double** aux = new double* [n];
    double** f = new double* [n];
    for (int i = 0;i < n;i++)
    {
        phi[i] = new double[n];
        aux[i] = new double[n];
        f[i] = new double[n];
    }
    for (int i = 0;i < n;i++)
    {
        for (int j = 0;j < n;j++)
        {
            phi[i][j] = 0;
            aux[i][j] = 0;
            f[i][j] = 1;
        }
    }
    double norm = compute_residual(phi, aux, f, n, comm);
    MPI_Barrier(comm);
    bool res = true;
    int jr, jc;
    if (rank == 0)
    {
        jr = 0;
        jc = 0;
    }
    else if (rank == 1)
    {
        jr = 0;
        jc = n - 1;
    }
    else if (rank == 2)
    {
        jr = n - 1;
        jc = n - 1;
    }
    else if (rank == 3)
    {
        jr = n - 1;
        jc = 0;
    }
    for (int i = 0;i < n;i++)
    {
        for (int j = 0;j < n;j++)
        {
            if (i == jr || j == jc)
            {
                continue;
            }
            if (aux[i][j] != 1)
            {
                res = false;
            }
        }
    }
    if (rank == 0)
    {
        printf("norm is: %5f\n", norm);
    }

    if (res == false)
    {
        printf("rank %d has error\n", rank);
    }
    for (int i = 0;i < n;i++)
    {
        delete[] phi[i];
        delete[] aux[i];
        delete[] f[i];
    }
    delete[] phi;
    delete[] aux;
    delete[] f;
    MPI_Finalize();

}