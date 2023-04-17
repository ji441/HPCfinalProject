#ifndef COMPUTE_RESIDUAL_H
#define COMPUTE_RESIDUAL_H

#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <cmath>
#include <iostream>
using namespace std;

//to DO, update the residual!
double compute_residual(double** phi, double** residual, double** f, int n, MPI_Comm comm)
{
    double norm = 0, totalnorm = 0;
    double h2 = pow(1.0 / ((double)n - 1.0), 2.0);
    int rank;
    MPI_Comm_rank(comm, &rank);
    double* rowbuf = new double[n];
    double* colbuf = new double[n];
    MPI_Status status;
    MPI_Request request_out1, request_in1, request_out3, request_in3;
    MPI_Request request_out2, request_in2, request_out4, request_in4;
    //initialize residual just in case
    for (int i = 0;i < n;i++)
    {
        for (int j = 0;j < n;j++)
        {
            residual[i][j] = 0;
        }
    }
    //nonblocking transfer data we need
    if (rank == 0) {//for this subpiece, we recv information
        MPI_Irecv(rowbuf, n, MPI_DOUBLE, 3, 1, comm, &request_in1);
        MPI_Irecv(colbuf, n, MPI_DOUBLE, 1, 1, comm, &request_in2);
    }
    else if (rank == 2) {//for this subpiece, we recv information
        MPI_Irecv(rowbuf, n, MPI_DOUBLE, 1, 1, comm, &request_in3);
        MPI_Irecv(colbuf, n, MPI_DOUBLE, 3, 1, comm, &request_in4);
    }
    else if (rank == 3) {
        //for this we send to rank 0 and 2
        //rowbuf = phi[1];
        //initialize the colbuf to send
        for (int i = 0;i < n;i++)
        {
            colbuf[i] = phi[i][n - 2];
            rowbuf[i] = phi[1][i];
        }
        MPI_Isend(rowbuf, n, MPI_DOUBLE, 0, 1, comm, &request_out1);
        MPI_Isend(colbuf, n, MPI_DOUBLE, 2, 1, comm, &request_out2);
    }
    else {
        //for this we still send
        //rowbuf = phi[n - 2];
        //initialize the colbuf to send
        for (int i = 0;i < n;i++)
        {
            colbuf[i] = phi[i][1];
            rowbuf[i] = phi[n - 2][i];
        }
        MPI_Isend(rowbuf, n, MPI_DOUBLE, 2, 1, comm, &request_out3);
        MPI_Isend(colbuf, n, MPI_DOUBLE, 0, 1, comm, &request_out4);
    }
    //for all subpiece, we calculate interior point first
    for (int i = 1;i < n - 1;i++)
    {
        for (int j = 1;j < n - 1;j++)
        {
            double res = f[i][j] + (phi[i][j - 1] + phi[i - 1][j] + phi[i + 1][j] + phi[i][j + 1]
                - 4.0 * phi[i][j]) / h2;
            residual[i][j] = res;
            norm += res * res;
        }
    }
    // since the middle 4 boundary is shared between 4 subpieces, we need to avoid recount
    if (rank == 0) {
        MPI_Wait(&request_in1, &status);
        MPI_Wait(&request_in2, &status);
        for (int k = 1;k < n - 1;k++) {
            //the right boundary index [k][n-1]
            double res1 = f[k][n - 1] + (phi[k][n - 2] + phi[k - 1][n - 1] + phi[k + 1][n - 1]
                + colbuf[k] - 4.0 * phi[k][n - 1]) / h2;
            //the upper boundary index [n-1][k]
            double res2 = f[n - 1][k] + (phi[n - 1][k - 1] + phi[n - 2][k] + phi[n - 1][k + 1]
                + rowbuf[k] - 4.0 * phi[k][n - 1]) / h2;
            //update residual
            residual[k][n - 1] = res1;
            residual[n - 1][k] = res2;
            norm += res1 * res1 + res2 * res2;
        }
        //only rank 0 count the error from middle point
        double resm = f[n - 1][n - 1] + (phi[n - 1][n - 2] + phi[n - 2][n - 1] + colbuf[n - 1] + rowbuf[n - 1]
            - 4.0 * phi[n - 1][n - 1]) / h2;
        norm += resm * resm;
        //update the middle point residual
        residual[n - 1][n - 1] = resm;
        //send the computed boundary residual to rank 1,3
        for (int k = 0;k < n;k++)
        {
            rowbuf[k] = residual[n - 1][k];
            colbuf[k] = residual[k][n - 1];
        }
        MPI_Send(rowbuf, n, MPI_DOUBLE, 3, 1, comm);
        MPI_Send(colbuf, n, MPI_DOUBLE, 1, 1, comm);


    }
    if (rank == 2) {
        MPI_Wait(&request_in3, &status);
        MPI_Wait(&request_in4, &status);
        for (int k = 1;k < n - 1;k++) {
            //the left boundary index [k][0]
            double res1 = f[k][0] + (phi[k][1] + phi[k - 1][0] + phi[k + 1][0]
                + colbuf[k] - 4.0 * phi[k][0]) / h2;
            //the bottom boundary index [0][k]
            double res2 = f[0][k] + (phi[0][k - 1] + phi[1][k] + phi[0][k + 1]
                + rowbuf[k] - 4.0 * phi[0][k]) / h2;
            //update residual
            residual[k][0] = res1;
            residual[0][k] = res2;
            norm += res1 * res1 + res2 * res2;
        }
        //calculate and update the middle point residual
        residual[0][0] = f[0][0] + (phi[1][0] + phi[0][1] + colbuf[0] + rowbuf[0] - 4 * phi[0][0]) / h2;
        //send the computed boundary residual to rank 1,3
        for (int k = 0;k < n;k++)
        {
            rowbuf[k] = residual[0][k];
            colbuf[k] = residual[k][0];
        }
        MPI_Send(rowbuf, n, MPI_DOUBLE, 1, 1, comm);
        MPI_Send(colbuf, n, MPI_DOUBLE, 3, 1, comm);

    }
    if (rank == 3) {
        MPI_Wait(&request_out1, &status);
        MPI_Wait(&request_out2, &status);
        MPI_Recv(rowbuf, n, MPI_DOUBLE, 0, 1, comm, &status);
        MPI_Recv(colbuf, n, MPI_DOUBLE, 2, 1, comm, &status);
        //update residual
        for (int k = 0;k < n;k++)
        {
            residual[0][k] = rowbuf[k];
            residual[k][n - 1] = colbuf[k];
        }

    }
    if (rank == 1) {
        MPI_Wait(&request_out3, &status);
        MPI_Wait(&request_out4, &status);
        MPI_Recv(rowbuf, n, MPI_DOUBLE, 2, 1, comm, &status);
        MPI_Recv(colbuf, n, MPI_DOUBLE, 0, 1, comm, &status);
        //update residual
        for (int k = 0;k < n;k++)
        {
            residual[n - 1][k] = rowbuf[k];
            residual[k][0] = colbuf[k];
        }
    }
    MPI_Allreduce(&norm, &totalnorm, 1, MPI_DOUBLE, MPI_SUM, comm);
    //clean up
    delete[] rowbuf;
    delete[] colbuf;
    return sqrt(totalnorm);

}

#endif