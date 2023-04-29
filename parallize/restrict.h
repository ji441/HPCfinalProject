#ifndef RESTRICT_H
#define RESTRICT_H
#include "compute_residual.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>

using namespace std;
void restrict(double ***phi, double ***f, double ***aux, int n, int level, MPI_Comm comm)
{
    // define MPI related things
    int rank;
    MPI_Comm_rank(comm, &rank);
    double *rowbuf = new double[n];
    double *colbuf = new double[n];
    double pointbuf;
    MPI_Status status;
    MPI_Request reqout1, reqout2, reqout3, reqin1, reqin2, reqin3;
    // we restrict the residual, so compute residual first
    compute_residual(phi[level], aux[level], f[level], n, comm);
    // printf("rank %d,compute residual completed\n", rank);
    MPI_Barrier(comm);
    // initialize the next level
    int n_coarse = (n - 1) / 2 + 1;
    int level_coarse = level + 1;
    int i_fine, j_fine;
    // create fbuf to besend
    double *frow = new double[n_coarse];
    double *fcol = new double[n_coarse];
    // initilize phi and f on next level
    for (int i = 0; i < n_coarse; i++)
    {
        for (int j = 0; j < n_coarse; j++)
        {
            phi[level_coarse][i][j] = 0.0;
            f[level_coarse][i][j] = 0.0;
        }
    }
    // communicate informations we need first,nonblocking version
    if (rank == 0)
    {
        // for this rank we recieve buf from rank 3 and 1
        // recv rowbuf from rank3
        MPI_Irecv(rowbuf, n, MPI_DOUBLE, 3, 1, comm, &reqin1);
        // recv colbuf from rank1
        MPI_Irecv(colbuf, n, MPI_DOUBLE, 1, 1, comm, &reqin2);
        // recv corner value from rank 2
        MPI_Irecv(&pointbuf, 1, MPI_DOUBLE, 2, 1, comm, &reqin3);
    }
    else if (rank == 1)
    {
        // for this block we send information to rank0 and rank2
        // initialize the col and row buf
        for (int i = 0; i < n; i++)
        {
            colbuf[i] = aux[level][i][1];
            rowbuf[i] = aux[level][n - 2][i];
        }
        // send col to rank 0
        MPI_Isend(colbuf, n, MPI_DOUBLE, 0, 1, comm, &reqout1);
        // send row to rank 2
        MPI_Isend(rowbuf, n, MPI_DOUBLE, 2, 1, comm, &reqout2);
    }
    else if (rank == 2)
    {
        // for this block we receive buf from rank 1 and 3
        // recv rowbuf from rank 1
        MPI_Irecv(rowbuf, n, MPI_DOUBLE, 1, 1, comm, &reqin1);
        // recv colbuf from rank 3
        MPI_Irecv(colbuf, n, MPI_DOUBLE, 3, 1, comm, &reqin2);
        // send to rank0 the corner value
        pointbuf = aux[level][1][1];
        MPI_Isend(&pointbuf, 1, MPI_DOUBLE, 0, 1, comm, &reqout1);
    }
    else
    {
        // for this block we send buf to rank 0 and 2
        // initialize the col and row buf to send
        for (int i = 0; i < n; i++)
        {
            rowbuf[i] = aux[level][1][i];
            colbuf[i] = aux[level][i][n - 2];
        }
        // send row to rank 0
        MPI_Isend(rowbuf, n, MPI_DOUBLE, 0, 1, comm, &reqout1);
        // send col to rank 2
        MPI_Isend(colbuf, n, MPI_DOUBLE, 2, 1, comm, &reqout2);
    }
    // while data is being transported, we compute interior points in each rank
    for (int i = 1; i < n_coarse - 1; i++)
    {
        for (int j = 1; j < n_coarse - 1; j++)
        {
            // get the correspoding finegrid index
            i_fine = (i * 2);
            j_fine = (j * 2);
            // perform restriction
            f[level_coarse][i][j] =
                aux[level][i_fine - 1][j_fine + 1] * (1.0 / 16.0) + aux[level][i_fine][j_fine + 1] * (1.0 / 8.0) +
                aux[level][i_fine + 1][j_fine + 1] * (1.0 / 16.0) + aux[level][i_fine - 1][j_fine] * (1.0 / 8.0) +
                aux[level][i_fine][j_fine] * (1.0 / 4.0) + aux[level][i_fine + 1][j_fine] * (1.0 / 8.0) +
                aux[level][i_fine - 1][j_fine - 1] * (1.0 / 16.0) + aux[level][i_fine][j_fine - 1] * (1.0 / 8.0) +
                aux[level][i_fine + 1][j_fine - 1] * (1.0 / 16.0);
        }
    }
    // now we deal with boundary problem
    if (rank == 0)
    {
        // printf("rank0 start receive\n");
        //  receive confirm
        MPI_Wait(&reqin1, &status);
        MPI_Wait(&reqin2, &status);
        MPI_Wait(&reqin3, &status);
        // printf("rank 0 information received\n");
        //  update on boundary
        for (int i = 1; i < n_coarse - 1; i++)
        {
            i_fine = (i * 2);
            // update upper doundary
            f[level_coarse][n_coarse - 1][i] =
                rowbuf[i_fine - 1] * (1.0 / 16.0) + rowbuf[i_fine] * (1.0 / 8.0) + rowbuf[i_fine + 1] * (1.0 / 16.0) +
                aux[level][n - 1][i_fine - 1] * (1.0 / 8.0) + aux[level][n - 1][i_fine] * (1.0 / 4.0) +
                aux[level][n - 1][i_fine + 1] * (1.0 / 8.0) + aux[level][n - 2][i_fine - 1] * (1.0 / 16.0) +
                aux[level][n - 2][i_fine] * (1.0 / 8.0) + aux[level][n - 2][i_fine + 1] * (1.0 / 16.0);
            // update right boundary
            f[level_coarse][i][n_coarse - 1] =
                colbuf[i_fine - 1] * (1.0 / 16.0) + colbuf[i_fine] * (1.0 / 8.0) + colbuf[i_fine + 1] * (1.0 / 16.0) +
                aux[level][i_fine - 1][n - 1] * (1.0 / 8.0) + aux[level][i_fine][n - 1] * (1.0 / 4.0) +
                aux[level][i_fine + 1][n - 1] * (1.0 / 8.0) + aux[level][i_fine - 1][n - 2] * (1.0 / 16.0) +
                aux[level][i_fine][n - 2] * (1.0 / 8.0) + aux[level][i_fine + 1][n - 2] * (1.0 / 16.0);
        }
        // update the rightup corner value, which is also the center value in all
        f[level_coarse][n_coarse - 1][n_coarse - 1] =
            rowbuf[n - 2] * (1.0 / 16.0) + rowbuf[n - 1] * (1.0 / 8.0) + pointbuf * (1.0 / 16.0) +
            aux[level][n - 1][n - 2] * (1.0 / 8.0) + aux[level][n - 1][n - 1] * (1.0 / 4.0) +
            colbuf[n - 1] * (1.0 / 8.0) + aux[level][n - 2][n - 2] * (1.0 / 16.0) +
            aux[level][n - 2][n - 1] * (1.0 / 8.0) + colbuf[n - 2] * (1.0 / 16.0);
        // initialize buf to send
        for (int i = 0; i < n_coarse; i++)
        {
            frow[i] = f[level_coarse][n_coarse - 1][i];
            fcol[i] = f[level_coarse][i][n_coarse - 1];
        }
        // send to rank 1, 2,3
        // send center value to rank 2
        pointbuf = f[level_coarse][n_coarse - 1][n_coarse - 1];
        MPI_Send(&pointbuf, 1, MPI_DOUBLE, 2, 1, comm);
        // send row to rank3
        MPI_Send(frow, n_coarse, MPI_DOUBLE, 3, 1, comm);
        // send col to rank1
        MPI_Send(fcol, n_coarse, MPI_DOUBLE, 1, 1, comm);
    }
    else if (rank == 1)
    {
        // printf("rank1 start receive\n");
        //  confirm data transported
        MPI_Wait(&reqout1, &status);
        MPI_Wait(&reqout2, &status);
        // receive f computed data from rank 0 and rank 2
        MPI_Recv(frow, n_coarse, MPI_DOUBLE, 2, 1, comm, &status);
        MPI_Recv(fcol, n_coarse, MPI_DOUBLE, 0, 1, comm, &status);
        // printf("rank 1 information received\n");
        //  write into f
        for (int k = 0; k < n_coarse; k++)
        {
            // write it into left most column
            f[level_coarse][k][0] = fcol[k];
            // write to upper boundary
            f[level_coarse][n_coarse - 1][k] = frow[k];
        }
    }
    else if (rank == 2)
    {
        // printf("rank2 start receive\n");
        //  message transport confirmation
        MPI_Wait(&reqin1, &status);
        MPI_Wait(&reqin2, &status);
        MPI_Wait(&reqout1, &status);
        // printf("rank 2 information received\n");
        //  update value on the boundary
        for (int k = 1; k < n_coarse - 1; k++)
        {
            int k_fine = k * 2;
            // update bottom row
            f[level_coarse][0][k] = rowbuf[k_fine - 1] * (1.0 / 16.0) + rowbuf[k_fine] * (1. / 8.) +
                                    rowbuf[k_fine + 1] * (1. / 16.) + aux[level][0][k_fine - 1] * (1. / 8.) +
                                    aux[level][0][k_fine] * (1. / 4.) + aux[level][0][k_fine + 1] * (1. / 8.) +
                                    aux[level][1][k_fine - 1] * (1. / 16.) + aux[level][1][k_fine] * (1. / 8.) +
                                    aux[level][1][k_fine + 1] * (1. / 16.);
            // update left most column
            f[level_coarse][k][0] = colbuf[k_fine - 1] * (1. / 16.) + colbuf[k_fine] * (1. / 8.) +
                                    colbuf[k_fine + 1] * (1. / 16.) + aux[level][k_fine - 1][0] * (1. / 8.) +
                                    aux[level][k_fine][0] * (1. / 4.) + aux[level][k_fine + 1][0] * (1. / 8.) +
                                    aux[level][k_fine - 1][1] * (1. / 16.) + aux[level][k_fine][1] * (1. / 8.) +
                                    aux[level][k_fine + 1][1] * (1. / 16.);
        }
        // for the corner value, recv from rank 0
        MPI_Recv(&pointbuf, 1, MPI_DOUBLE, 0, 1, comm, &status);
        f[level_coarse][0][0] = pointbuf;
        // initialize buf to send
        for (int k = 0; k < n_coarse; k++)
        {
            frow[k] = f[level_coarse][0][k];
            fcol[k] = f[level_coarse][k][0];
        }
        // send to rank 1 and 3
        MPI_Send(frow, n_coarse, MPI_DOUBLE, 1, 1, comm);
        MPI_Send(fcol, n_coarse, MPI_DOUBLE, 3, 1, comm);
    }
    else
    {
        // printf("rank3 start receive\n");
        MPI_Wait(&reqout1, &status);
        MPI_Wait(&reqout2, &status);
        // printf("rank 3 information received\n");
        //  get data from rank 0,2
        MPI_Recv(frow, n_coarse, MPI_DOUBLE, 0, 1, comm, &status);
        MPI_Recv(fcol, n_coarse, MPI_DOUBLE, 2, 1, comm, &status);
        // write into f
        for (int k = 0; k < n_coarse; k++)
        {
            // write row 0
            f[level_coarse][0][k] = frow[k];
            // write col n_coarse -1
            f[level_coarse][k][n_coarse - 1] = fcol[k];
        }
    }
    // freesapces
    delete[] frow, rowbuf;
    delete[] fcol, colbuf;
}

#endif
