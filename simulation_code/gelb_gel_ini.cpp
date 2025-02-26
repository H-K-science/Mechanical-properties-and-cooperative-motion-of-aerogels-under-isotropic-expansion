#include <iostream>
#include <iomanip>
#include <omp.h>
#include <time.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <stack>
#include <queue>
#include <sstream>
#include <algorithm>
#include <utility>
using namespace std;

#define PI 3.1415926535
/*-----  INPUT filename  -----*/
char filename_input[64];
const char *name_input_p = "INPUT_particle_02020";
const char *name_input_b = "INPUT_bond_02020";

/*-----  SYSTEM SIZE  -----*/
const double L = 76.4045;
/*-----  Number of Particle  -----*/
const int N = 10648;
/*-----  Bonding probability  -----*/
const double p_bond = 1.0;
const double d_r = 0.005;
/*-----  Hardcore potential parameter  -----*/
const double e = 2.5;    // Depth [kJ/mol]
const double nn_r = 6.0; // nn_list range

/*-----  Temperature[K]  -----*/
const double T = 300;
const double kBT = 2.5;
/*-----  time[ps] -----*/
const double dt = 0.01;
/*-----  translational friction coefficient[ps^(-1)]  -----*/
const double gamma_t = 1.82;
/*-----  Mass and Moment of Inertia  -----*/
const double m = 5547;
const double I = 2219;

/*-----  Bonding potential parameter  -----*/
const double e_bond = 60.0;
const double k_bond = 13.6377;
const double s_equi = 0.3;
const double s_max = 0.5;
const double k_angle = 1200.0;
const double k_tort = 1200.0;

/*-----  Other parameters  -----*/
const double gauss_cut = 4.0;
const double er_int = 1.0e-4; // inv_r2's error
const double d_q = 1.0e-2; // check sin_theta

/*-----  Unspecified parameters -----*/
double tau_B = m * gamma_t / kBT;
const double ndens = (double)N / L / L / L;
const double phi = 4.0 * PI * (double)N / 3 / L / L / L;
const double nn_r2 = nn_r * nn_r;
double Etot, U, K;
int N_bond;

/*-----  OUTPUT  -----*/
char filename[64];
char filename_temperature[128];

const int output_display = 400;
const int write_file = 400;
const char *name_p = "particle";
const char *name_b = "bond";

const int out_temperature = 400;
const char *name_temperature = "tmp_00000";

/*------------------------------------------------------*/
/*----------------------  FUNCTION  --------------------*/
/*------------------------------------------------------*/
double inverse(double a, double e) // e: error
{
    if (!(abs(a) > e))
        return 0;
    else
        return 1 / a;
} // for inverse
void Dist_PBC(vector<double> &R) // jufge =0->Dist or 1->Pos
{
    for (int i = 0; i < 3; i++)
    {
        if (R[i] >= 0.5 * L)
            R[i] -= L;
        if (R[i] <= -0.5 * L)
            R[i] += L;
    }
}
void Dist_PBC_parallel(double R[]) // jufge =0->Dist or 1->Pos
{
    for (int i = 0; i < 3; i++)
    {
        if (R[i] >= 0.5 * L)
            R[i] -= L;
        if (R[i] <= -0.5 * L)
            R[i] += L;
    }
}
double Gauss(double var)
{
    double R1, R2, R;
    R1 = (double)rand() / RAND_MAX;
    R2 = (double)rand() / RAND_MAX;
    R = sqrt(-2 * var * log(R1)) * sin(2 * PI * R2);
    if (!(abs(R) < gauss_cut * sqrt(var)))
        R = 0.0;

    return R;
}
double C(double y) // y = 0.5 * gamma_t * dt
{
    return pow(y, 3) * 2.0 / 3.0 - pow(y, 4) * 0.5 + pow(y, 5) * 7.0 / 30.0 - pow(y, 6) / 12.0 + pow(y, 7) * 31.0 / 1260.0 - pow(y, 8) / 160.0 + pow(y, 9) * 127.0 / 90720.0;
}
double BoverC(double y)
{
    return y * 0.5 + pow(y, 2) * 7.0 / 8.0 + pow(y, 3) * 367.0 / 480.0 - pow(y, 4) * 857.0 / 1920.0 + pow(y, 5) * 52813.0 / 268800.0 - pow(y, 6) * 224881.0 / 3225600.0 + pow(y, 7) * 1341523.0 / 64512000.0;
}
double one_minus_exp(double x) // x = gamma_t * dt
{
    return x - pow(x, 2) * 0.5 + pow(x, 3) / 6.0 - pow(x, 4) / 24.0 + pow(x, 5) / 120.0 - pow(x, 6) / 720.0 + pow(x, 7) / 5040.0;
}
double Bover_exp_minus(double y)
{
    return -pow(y, 3) / 6.0 + pow(y, 5) / 60.0 - pow(y, 7) * 17.0 / 10080.0 - pow(y, 9) * 31.0 / 181440;
}
double yDoverC(double y)
{
    return -1.5 - y * 9.0 / 8.0 - pow(y, 2) * 71.0 / 160.0 - pow(y, 3) * 81.0 / 640.0 - pow(y, 4) * 7807.0 / 268800.0 - pow(y, 5) * 1971.0 / 358400.0 - pow(y, 6) * 56417.0 / 64512000.0;
}
double D_exp_minus_one(double y)
{
    return y * 0.5 + pow(y, 2) * 0.5 + pow(y, 3) * 5.0 / 24.0 + pow(y, 4) * 1.0 / 24.0 + pow(y, 5) * 1.0 / 240.0 + pow(y, 6) * 1.0 / 720.0 + pow(y, 7) * 5.0 / 8064.0;
}
double exp_exp(double y)
{
    return y * 2.0 + pow(y, 3) / 3.0 + pow(y, 5) / 60.0 + pow(y, 7) / 2520.0;
}

void conv_r_to_p(double &Lx, double &Ly, double &Lz, double q1, double q2, double q3, double q4)
{
    vector<double> R(9);
    R[0] = -q1 * q1 + q2 * q2 - q3 * q3 + q4 * q4;
    R[1] = 2 * (q3 * q4 - q1 * q2);
    R[2] = 2 * (q2 * q3 + q1 * q4);
    R[3] = -2 * (q1 * q2 + q3 * q4);
    R[4] = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
    R[5] = 2 * (q2 * q4 - q1 * q3);
    R[6] = 2 * (q2 * q3 - q1 * q4);
    R[7] = -2 * (q1 * q3 + q2 * q4);
    R[8] = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4;

    double Lx_r, Ly_r, Lz_r;
    Lx_r = Lx;
    Ly_r = Ly;
    Lz_r = Lz;

    Lx = R[0] * Lx_r + R[1] * Ly_r + R[2] * Lz_r;
    Ly = R[3] * Lx_r + R[4] * Ly_r + R[5] * Lz_r;
    Lz = R[6] * Lx_r + R[7] * Ly_r + R[8] * Lz_r;
}

vector<double> vec_p_to_r(double &Lx, double &Ly, double &Lz, double q1, double q2, double q3, double q4)
{
    vector<double> R(9);
    R[0] = -q1 * q1 + q2 * q2 - q3 * q3 + q4 * q4;
    R[1] = 2 * (q3 * q4 - q1 * q2);
    R[2] = 2 * (q2 * q3 + q1 * q4);
    R[3] = -2 * (q1 * q2 + q3 * q4);
    R[4] = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
    R[5] = 2 * (q2 * q4 - q1 * q3);
    R[6] = 2 * (q2 * q3 - q1 * q4);
    R[7] = -2 * (q1 * q3 + q2 * q4);
    R[8] = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4;

    vector<double> Lr(3);

    Lr[0] = R[0] * Lx + R[3] * Ly + R[6] * Lz;
    Lr[1] = R[1] * Lx + R[4] * Ly + R[7] * Lz;
    Lr[2] = R[2] * Lx + R[5] * Ly + R[8] * Lz;

    return Lr;
}
vector<double> vec_r_to_p(double &Lx, double &Ly, double &Lz, double q1, double q2, double q3, double q4)
{
    vector<double> R(9);
    R[0] = -q1 * q1 + q2 * q2 - q3 * q3 + q4 * q4;
    R[1] = 2 * (q3 * q4 - q1 * q2);
    R[2] = 2 * (q2 * q3 + q1 * q4);
    R[3] = -2 * (q1 * q2 + q3 * q4);
    R[4] = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
    R[5] = 2 * (q2 * q4 - q1 * q3);
    R[6] = 2 * (q2 * q3 - q1 * q4);
    R[7] = -2 * (q1 * q3 + q2 * q4);
    R[8] = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4;

    vector<double> Lp(3);

    Lp[0] = R[0] * Lx + R[1] * Ly + R[2] * Lz;
    Lp[1] = R[3] * Lx + R[4] * Ly + R[5] * Lz;
    Lp[2] = R[6] * Lx + R[7] * Ly + R[8] * Lz;

    return Lp;
}

double calc_LJ_Force(vector<double> &rad, vector<double> &Rx, vector<double> &Ry, vector<double> &Rz, vector<double> &f_int,
                     vector<double> &Fx, vector<double> &Fy, vector<double> &Fz, vector<vector<int> > &nn_list)
{
    double fx, fy, fz, r2, r_min2, sigma_12, inv_r2, u, U = 0.0;
    double D[3];
    int p1, p2;

#pragma omp parallel for private(p2, D, r2, r_min2, sigma_12, inv_r2, u) reduction(+ \
                                                                                   : U) schedule(static, 10)
    for (p1 = 0; p1 < N - 1; p1++)
    {
        for (p2 = p1 + 1; p2 < N; p2++)
        {
            D[0] = Rx[p1] - Rx[p2];
            D[1] = Ry[p1] - Ry[p2];
            D[2] = Rz[p1] - Rz[p2];
            Dist_PBC_parallel(D);

            r2 = D[0] * D[0] + D[1] * D[1] + D[2] * D[2];
            if (r2 <= nn_r2) // LJ
            {
                nn_list[p1].push_back(p2); // count in nn_list using r_cut
                r_min2 = (rad[p1] + rad[p2] + s_equi) * (rad[p1] + rad[p2] + s_equi);
                f_int[p1 * N * 3 + p2 * 3 + 0] = 0.0;
                f_int[p1 * N * 3 + p2 * 3 + 1] = 0.0;
                f_int[p1 * N * 3 + p2 * 3 + 2] = 0.0;

                if (r2 < r_min2)
                {
                    sigma_12 = (rad[p1] + rad[p2] + s_equi) / 1.122462;
                    sigma_12 = sigma_12 * sigma_12;
                    inv_r2 = 1 / r2;
                    u = 48 * e * (pow(sigma_12 * inv_r2, 3) - 0.5) * pow(sigma_12 * inv_r2, 3);
                    U += 4 * e * (pow(sigma_12 * inv_r2, 3) - 1) * pow(sigma_12 * inv_r2, 3);
                    f_int[p1 * N * 3 + p2 * 3 + 0] = u * D[0] * inv_r2;
                    f_int[p1 * N * 3 + p2 * 3 + 1] = u * D[1] * inv_r2;
                    f_int[p1 * N * 3 + p2 * 3 + 2] = u * D[2] * inv_r2;
                }
            }

        } // particle 2
    }     // particle 1

    for (int p1 = 0; p1 < N - 1; p1++)
    {
        for (int i = 0; i < nn_list[p1].size(); i++)
        {
            int p2 = nn_list[p1][i];
            fx = f_int[p1 * N * 3 + p2 * 3 + 0];
            fy = f_int[p1 * N * 3 + p2 * 3 + 1];
            fz = f_int[p1 * N * 3 + p2 * 3 + 2];
            Fx[p1] += fx;
            Fy[p1] += fy;
            Fz[p1] += fz;
            Fx[p2] -= fx;
            Fy[p2] -= fy;
            Fz[p2] -= fz;
        }
    }

    return U;
}

double diff_theta(vector<double> &C, vector<double> &vp, double q1, double q2, double q3, double q4)
{
    double R13, R23, R31, R32, R33, sin_theta;
    R13 = 2.0 * (q2 * q3 + q1 * q4);
    R23 = 2.0 * (q2 * q4 - q1 * q3);
    R31 = 2.0 * (q2 * q3 - q1 * q4);
    R32 = -2.0 * (q1 * q3 + q2 * q4);
    R33 = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4;
    sin_theta = sqrt(R31 * R31 + R32 * R32);

    vector<double> RC(9);
    RC[0] = R31 * R13 / sin_theta;
    RC[1] = R31 * R23 / sin_theta;
    RC[2] = R33 * R31 / sin_theta;
    RC[3] = -1.0 * R32 * R13 / sin_theta;
    RC[4] = -1.0 * R32 * R23 / sin_theta;
    RC[5] = -1.0 * R33 * R32 / sin_theta;
    RC[6] = R33 * R13 / sin_theta;
    RC[7] = R33 * R23 / sin_theta;
    RC[8] = sin_theta;

    return C[0] * (RC[0] * vp[0] + RC[1] * vp[1] + RC[2] * vp[2]) - C[1] * (RC[3] * vp[0] + RC[4] * vp[1] + RC[5] * vp[2]) + C[2] * (RC[6] * vp[0] + RC[7] * vp[1] - RC[8] * vp[2]);
}

void Torq_r(double &tx_r, double &ty_r, double &tz_r, double f_phi, double f_theta, double f_psi, double q1, double q2, double q3, double q4)
{
    double R31, R32, R33, sin_2_theta, sin_theta;
    R31 = 2.0 * (q2 * q3 - q1 * q4);
    R32 = -2.0 * (q1 * q3 + q2 * q4);
    R33 = -q1 * q1 - q2 * q2 + q3 * q3 + q4 * q4;
    sin_2_theta = R31 * R31 + R32 * R32;
    sin_theta = sqrt(R31 * R31 + R32 * R32);

    vector<double> RP(6);
    RP[0] = -1.0 * R33 * R31 / sin_2_theta;
    RP[1] = -1.0 * R32 / sin_theta;
    RP[2] = R31 / sin_2_theta;
    RP[3] = -1.0 * R33 * R32 / sin_2_theta;
    RP[4] = R31 / sin_theta;
    RP[5] = R32 / sin_2_theta;

    tx_r = RP[0] * f_phi + RP[1] * f_theta + RP[2] * f_psi;
    ty_r = RP[3] * f_phi + RP[4] * f_theta + RP[5] * f_psi;
    tz_r = f_phi;
}

double calc_Bond_Force(vector<double> &rad, vector<double> &Rx, vector<double> &Ry, vector<double> &Rz,
                       vector<double> &q1, vector<double> &q2, vector<double> &q3, vector<double> &q4,
                       vector<vector<int> > &Bond, vector<double> &s_bond, vector<double> &t_bond,
                       vector<double> &Fx, vector<double> &Fy, vector<double> &Fz,
                       vector<double> &Tx, vector<double> &Ty, vector<double> &Tz)
{
    double U;

    int p1, p2;
    double fx, fy, fz, tx, ty, tz;
    vector<double> s12_r(3), s12_r_norm(3), s1_p(3), s2_p(3), s1_r(3), s2_r(3), t1_p(3), t2_p(3), t1_r(3), t2_r(3), s12_p_other(3), s_p_other(3), t_p_other(3);
    double fs, fa, ft, f_phi, f_theta, f_psi, P, Q;
    double s12, exp_morse, exp_cut, cos_12, cos_21, d_cos_theta, d_cos_phi;
    double norm, sin_theta;

    for (p1 = 0; p1 < N - 1; p1++)
    {
        for (int i = 0; i < Bond[p1].size(); i++)
        {
            p2 = Bond[p1][i];

            s1_p[0] = s_bond[p1 * N * 3 + p2 * 3 + 0];
            s1_p[1] = s_bond[p1 * N * 3 + p2 * 3 + 1];
            s1_p[2] = s_bond[p1 * N * 3 + p2 * 3 + 2];
            t1_p[0] = t_bond[p1 * N * 3 + p2 * 3 + 0];
            t1_p[1] = t_bond[p1 * N * 3 + p2 * 3 + 1];
            t1_p[2] = t_bond[p1 * N * 3 + p2 * 3 + 2];
            s1_r = vec_p_to_r(s1_p[0], s1_p[1], s1_p[2], q1[p1], q2[p1], q3[p1], q4[p1]);
            t1_r = vec_p_to_r(t1_p[0], t1_p[1], t1_p[2], q1[p1], q2[p1], q3[p1], q4[p1]);
            sin_theta = (q2[p1] * q3[p1] + q1[p1] * q4[p1]) * (q2[p1] * q3[p1] + q1[p1] * q4[p1]);
            sin_theta += (q2[p1] * q4[p1] - q1[p1] * q3[p1]) * (q2[p1] * q4[p1] - q1[p1] * q3[p1]);
            sin_theta = 2.0 * sqrt(sin_theta);
            if (sin_theta < d_q)
            {
                cout << "!!!\n";
                q1[p1] = 0.0;
                q2[p1] = 1.0 / sqrt(2.0);
                q3[p1] = 0.0;
                q4[p1] = 1.0 / sqrt(2.0);
                s1_p = vec_p_to_r(s1_r[0], s1_r[1], s1_r[2], q1[p1], q2[p1], q3[p1], q4[p1]);
                t1_p = vec_p_to_r(t1_r[0], t1_r[1], t1_r[2], q1[p1], q2[p1], q3[p1], q4[p1]);
                s_bond[p1 * N * 3 + p2 * 3 + 0] = s1_p[0];
                s_bond[p1 * N * 3 + p2 * 3 + 1] = s1_p[1];
                s_bond[p1 * N * 3 + p2 * 3 + 2] = s1_p[2];
                t_bond[p1 * N * 3 + p2 * 3 + 0] = t1_p[0];
                t_bond[p1 * N * 3 + p2 * 3 + 1] = t1_p[1];
                t_bond[p1 * N * 3 + p2 * 3 + 2] = t1_p[2];
            }

            s2_p[0] = s_bond[p2 * N * 3 + p1 * 3 + 0];
            s2_p[1] = s_bond[p2 * N * 3 + p1 * 3 + 1];
            s2_p[2] = s_bond[p2 * N * 3 + p1 * 3 + 2];
            t2_p[0] = t_bond[p2 * N * 3 + p1 * 3 + 0];
            t2_p[1] = t_bond[p2 * N * 3 + p1 * 3 + 1];
            t2_p[2] = t_bond[p2 * N * 3 + p1 * 3 + 2];
            s2_r = vec_p_to_r(s2_p[0], s2_p[1], s2_p[2], q1[p2], q2[p2], q3[p2], q4[p2]);
            t2_r = vec_p_to_r(t2_p[0], t2_p[1], t2_p[2], q1[p2], q2[p2], q3[p2], q4[p2]);
            sin_theta = (q2[p2] * q3[p2] + q1[p2] * q4[p2]) * (q2[p2] * q3[p2] + q1[p2] * q4[p2]);
            sin_theta += (q2[p2] * q4[p2] - q1[p2] * q3[p2]) * (q2[p2] * q4[p2] - q1[p2] * q3[p2]);
            sin_theta = 2.0 * sqrt(sin_theta);
            if (sin_theta < d_q)
            {
                cout << "!!!\n";
                q1[p2] = 0.0;
                q2[p2] = 1.0 / sqrt(2.0);
                q3[p2] = 0.0;
                q4[p2] = 1.0 / sqrt(2.0);
                s2_p = vec_p_to_r(s2_r[0], s2_r[1], s2_r[2], q1[p2], q2[p2], q3[p2], q4[p2]);
                t2_p = vec_p_to_r(t2_r[0], t2_r[1], t2_r[2], q1[p2], q2[p2], q3[p2], q4[p2]);
                s_bond[p2 * N * 3 + p1 * 3 + 0] = s2_p[0];
                s_bond[p2 * N * 3 + p1 * 3 + 1] = s2_p[1];
                s_bond[p2 * N * 3 + p1 * 3 + 2] = s2_p[2];
                t_bond[p2 * N * 3 + p1 * 3 + 0] = t2_p[0];
                t_bond[p2 * N * 3 + p1 * 3 + 1] = t2_p[1];
                t_bond[p2 * N * 3 + p1 * 3 + 2] = t2_p[2];
            }

            s12_r[0] = (Rx[p2] + rad[p2] * s2_r[0]) - (Rx[p1] + rad[p1] * s1_r[0]);
            s12_r[1] = (Ry[p2] + rad[p2] * s2_r[1]) - (Ry[p1] + rad[p1] * s1_r[1]);
            s12_r[2] = (Rz[p2] + rad[p2] * s2_r[2]) - (Rz[p1] + rad[p1] * s1_r[2]);
            Dist_PBC(s12_r);

            s12 = sqrt(s12_r[0] * s12_r[0] + s12_r[1] * s12_r[1] + s12_r[2] * s12_r[2]);
            s12_r_norm[0] = s12_r[0] / s12;
            s12_r_norm[1] = s12_r[1] / s12;
            s12_r_norm[2] = s12_r[2] / s12;

            cos_12 = -1.0 * (s12_r_norm[0] * s1_r[0] + s12_r_norm[1] * s1_r[1] + s12_r_norm[2] * s1_r[2]);
            cos_21 = (s12_r_norm[0] * s2_r[0] + s12_r_norm[1] * s2_r[1] + s12_r_norm[2] * s2_r[2]);
            d_cos_theta = 2.0 + cos_12 + cos_21;
            d_cos_phi = 1.0 + t1_r[0] * t2_r[0] + t1_r[1] * t2_r[1] + t1_r[2] * t2_r[2];

            exp_morse = exp(-k_bond * (s12 - s_equi));
            fs = 2.0 * e_bond * k_bond * (1.0 - exp_morse) * exp_morse / s12;
            exp_cut = 1.0;
            fa = 0.0;
            ft = 0.0;
            

            fx = (fs + fa + ft) * (-s12_r[0]) + k_angle * exp_cut * ((-s12_r_norm[1] * s12_r_norm[1] - s12_r_norm[2] * s12_r_norm[2]) * (s2_r[0] - s1_r[0]) + (s12_r_norm[0] * s12_r_norm[1]) * (s2_r[1] - s1_r[1]) + (s12_r_norm[0] * s12_r_norm[2]) * (s2_r[2] - s1_r[2])) / s12;
            fx *= -1.0;
            fy = (fs + fa + ft) * (-s12_r[1]) + k_angle * exp_cut * ((s12_r_norm[0] * s12_r_norm[1]) * (s2_r[0] - s1_r[0]) + (-s12_r_norm[0] * s12_r_norm[0] - s12_r_norm[2] * s12_r_norm[2]) * (s2_r[1] - s1_r[1]) + (s12_r_norm[1] * s12_r_norm[2]) * (s2_r[2] - s1_r[2])) / s12;
            fy *= -1.0;
            fz = (fs + fa + ft) * (-s12_r[2]) + k_angle * exp_cut * ((s12_r_norm[2] * s12_r_norm[0]) * (s2_r[0] - s1_r[0]) + (s12_r_norm[2] * s12_r_norm[1]) * (s2_r[1] - s1_r[1]) + (-s12_r_norm[0] * s12_r_norm[0] - s12_r_norm[1] * s12_r_norm[1]) * (s2_r[2] - s1_r[2])) / s12;
            fz *= -1.0;

            Fx[p1] += fx;
            Fx[p2] -= fx;

            Fy[p1] += fy;
            Fy[p2] -= fy;

            Fz[p1] += fz;
            Fz[p2] -= fz;

            // Torque for p1
            P = -1.0 * (fs + fa + ft) + k_angle * exp_cut * ((cos_21 + cos_12) / s12 - 1.0 / rad[p1] - 1.0 / rad[p2]) / s12;
            Q = k_tort * exp_cut;

            f_phi = P * (s1_r[0] * s12_r[1] - s1_r[1] * s12_r[0]) * rad[p1] + Q * (t1_r[0] * t2_r[1] - t1_r[1] * t2_r[0]);
            f_phi *= -1.0;

            s12_p_other = vec_r_to_p(s12_r[0], s12_r[1], s12_r[2], q1[p1], q2[p1], q3[p1], q4[p1]); // s12_p on p1
            t_p_other = vec_r_to_p(t2_r[0], t2_r[1], t2_r[2], q1[p1], q2[p1], q3[p1], q4[p1]);      // t2_p on p1
            f_psi = P * (s1_p[0] * s12_p_other[1] - s1_p[1] * s12_p_other[0]) * rad[p1] + Q * (t1_p[0] * t_p_other[1] - t1_p[1] * t_p_other[0]);
            f_psi *= -1.0;

            f_theta = P * diff_theta(s12_r, s1_p, q1[p1], q2[p1], q3[p1], q4[p1]) * rad[p1] + Q * diff_theta(t2_r, t1_p, q1[p1], q2[p1], q3[p1], q4[p1]);
            f_theta *= -1.0;

            Torq_r(tx, ty, tz, f_phi, f_theta, f_psi, q1[p1], q2[p1], q3[p1], q4[p1]);
            Tx[p1] += tx;
            Ty[p1] += ty;
            Tz[p1] += tz;

            // Torque for p2
            s12_r[0] *= -1.0;
            s12_r[1] *= -1.0;
            s12_r[2] *= -1.0; // convert s12_r to s21_r

            f_phi = P * (s2_r[0] * s12_r[1] - s2_r[1] * s12_r[0]) * rad[p2] + Q * (t2_r[0] * t1_r[1] - t2_r[1] * t1_r[0]);
            f_phi *= -1.0;

            s12_p_other = vec_r_to_p(s12_r[0], s12_r[1], s12_r[2], q1[p2], q2[p2], q3[p2], q4[p2]); // s21_p on p2
            t_p_other = vec_r_to_p(t1_r[0], t1_r[1], t1_r[2], q1[p2], q2[p2], q3[p2], q4[p2]);      // t1_p on p2
            f_psi = P * (s2_p[0] * s12_p_other[1] - s2_p[1] * s12_p_other[0]) * rad[p2] + Q * (t2_p[0] * t_p_other[1] - t2_p[1] * t_p_other[0]);
            f_psi *= -1.0;

            f_theta = P * diff_theta(s12_r, s2_p, q1[p2], q2[p2], q3[p2], q4[p2]) * rad[p2] + Q * diff_theta(t1_r, t2_p, q1[p2], q2[p2], q3[p2], q4[p2]);
            f_theta *= -1.0;

            Torq_r(tx, ty, tz, f_phi, f_theta, f_psi, q1[p2], q2[p2], q3[p2], q4[p2]);
            Tx[p2] += tx;
            Ty[p2] += ty;
            Tz[p2] += tz;

            U += e_bond * ((1.0 - exp_morse) * (1.0 - exp_morse) - 1.0) + k_angle * d_cos_theta * exp_cut + k_tort * d_cos_phi * exp_cut;
        }
    }
    return U;
}

int calc_Nc(vector<vector<int> > &Bond, int &N_bond)
{
    vector<bool> seen(N, false); // 既に見たことがある頂点か記録する
    int i, state, next;
    int Nc = 0;
    N_bond = 0;

    vector<vector<int> > Bond_both(N, vector<int>());
    int p1, p2;
    for (p1 = 0; p1 < N - 1; p1++)
    {
        for (i = 0; i < Bond[p1].size(); i++)
        {
            p2 = Bond[p1][i];
            Bond_both[p1].push_back(p2);
            Bond_both[p2].push_back(p1);
        }
    }

    int p;
    for (p = 0; p < N; p++)
    {
        N_bond += Bond_both[p].size();
        if (!seen[p])
        {
            Nc += 1;
            stack<int> st;
            st.push(p); // pから探索する
            seen[p] = true;

            while (st.size() != 0)
            {                     // 深さ優先探索
                state = st.top(); // 現在の状態
                st.pop();
                for (i = 0; i < Bond_both[state].size(); i++)
                {
                    next = Bond_both[state][i];
                    if (!seen[next])
                    {                  // 未探索の時のみ行う
                        st.push(next); //次の状態をqueueへ格納
                        seen[next] = true;
                    }
                }
            }
        }
    }

    return Nc;
}

void WRITE_parameter_init(vector<double> &rad, vector<double> &Rx, vector<double> &Ry, vector<double> &Rz, vector<double> &Vx, vector<double> &Vy, vector<double> &Vz,
                          vector<double> &X_plus_x_t, vector<double> &X_plus_y_t, vector<double> &X_plus_z_t,
                          vector<double> &q1, vector<double> &q2, vector<double> &q3, vector<double> &q4,
                          vector<double> &Omega_x, vector<double> &Omega_y, vector<double> &Omega_z,
                          vector<double> &X_plus_x_r, vector<double> &X_plus_y_r, vector<double> &X_plus_z_r)
{
    ofstream ofs;

    sprintf(filename, "A_PARAMETER_INITIAL_CONDITION_ini.dat");
    ofs.open(filename);
    ofs << "---------------------------------------------------\n";
    ofs << "--------------------- Gelb Model ------------------\n";
    ofs << "--------------------- Leap-frog -------------------\n";
    ofs << "---------------------------------------------------\n\n\n\n";

    ofs << "------------------  Fundamental PARAMETER  -----------------\n";
    ofs << setw(10) << "L = " << setw(10) << L << ": Size of Simulation Box(cubic) \n";
    ofs << setw(10) << "N = " << setw(10) << N << ": Number of Particles \n";
    ofs << setw(10) << "p_bond = " << setw(10) << p_bond << ": Bonding probability \n";
    ofs << setw(10) << "nn_r = " << setw(10) << nn_r << ": bonding check range \n";
    ofs << setw(10) << "T = " << setw(10) << T << ": Temperature[K] \n";
    ofs << setw(10) << "kBT = " << setw(10) << kBT << ": kBT[kJ/mol] \n";
    ofs << setw(10) << "dt = " << setw(10) << dt << ": Time Step[ps] \n";
    ofs << setw(10) << "gamma_t = " << setw(10) << gamma_t << ": Translational collision Coefficient[ps^(-1)] \n";
    ofs << setw(10) << "m = " << setw(10) << m << ": standard mass\n";
    ofs << setw(10) << "I = " << setw(10) << I << ": standard moment of inertia \n\n\n";

    ofs << "------------------  Hardcore potential  -----------------\n";
    ofs << setw(10) << "WCA potential\n";
    ofs << setw(10) << "e = " << setw(10) << e << ": Depth of Energy \n\n\n";

    ofs << "------------------  Bonding potential  -----------------\n";
    ofs << setw(10) << "e_bond = " << setw(10) << e_bond << ": Depth of Morse potential \n";
    ofs << setw(10) << "k_bond = " << setw(10) << k_bond << ": Strength of Morse potential \n";
    ofs << setw(10) << "s_equi = " << setw(10) << s_equi << ": Equilibrium length of bond \n";
    ofs << setw(10) << "s_max = " << setw(10) << s_max << ": Maximum lenggth of bond \n";
    ofs << setw(10) << "k_angle = " << setw(10) << k_angle << ": Strength of angle potential \n";
    ofs << setw(10) << "k_tort = " << setw(10) << k_tort << ": Strength of tortional potential \n\n\n";

    ofs << "------------------  Other parameters  -----------------\n";
    ofs << setw(10) << "gauss_cut = " << setw(10) << gauss_cut << ": cutoff of gauss distribution\n";
    ofs << setw(10) << "er_int = " << setw(10) << er_int << ": Error of Inverse of r2\n";
    ofs << setw(10) << "d_q = " << setw(10) << d_q << ": Tolerance of sin_theta aroun zero\n\n\n";

    ofs << "------------------  Unspecified parameters  -----------------\n";
    ofs << setw(10) << "tau_B = " << setw(10) << tau_B << ": Brownian time [ps] \n";
    ofs << setw(10) << "ndens = " << setw(10) << ndens << ": Number Density of Particles \n";
    ofs << setw(10) << "phi = " << setw(10) << phi << ": Volume Fraction of Particles \n\n\n\n";

    ofs << "------------------  OUTPUT FILE  -----------------\n";
    ofs << "file output step = " << write_file << '\n';
    ofs << "particle_00000.dat\n";
    ofs << "number, radius, Rx, Ry, Rz, Vx, Vy, Vz, X_plus_x_t, X_plus_y_t, X_plus_z_t, q1, q2,q3, q4, Omega_x, Omega_y, Omega_z, X_plus_x_r, X_plus_y_r, X_plus_z_r\n\n\n\n\n\n";

    ofs << "--------------  INITIAL CONDITION  ----------------\n";
    ofs << "number, radius, Rx, Ry, Rz, Vx, Vy, Vz, X_plus_x_t, X_plus_y_t, X_plus_z_t, q1, q2,q3, q4, Omega_x, Omega_y, Omega_z, X_plus_x_r, X_plus_y_r, X_plus_z_r\n";
    for (int i = 0; i < N; i++)
    {
        ofs << setw(5) << i << "," << setw(10) << rad[i] << "," << setw(10) << Rx[i] << "," << setw(10) << Ry[i] << "," << setw(10) << Rz[i] << ",";
        ofs << setw(10) << Vx[i] << "," << setw(10) << Vy[i] << "," << setw(10) << Vz[i] << ",";
        ofs << setw(10) << X_plus_x_t[i] << "," << setw(10) << X_plus_y_t[i] << "," << setw(10) << X_plus_z_t[i] << ",";
        ofs << setw(10) << q1[i] << "," << setw(10) << q2[i] << "," << setw(10) << q3[i] << "," << setw(10) << q4[i] << ",";
        ofs << setw(10) << Omega_x[i] << "," << setw(10) << Omega_y[i] << "," << setw(10) << Omega_z[i] << ",";
        ofs << setw(10) << X_plus_x_r[i] << "," << setw(10) << X_plus_y_r[i] << "," << setw(10) << X_plus_z_r[i] << ",\n";
    }
    ofs.close();
}

/*------------------------------------------------------*/
/*----------------------  MAIN  ------------------------*/
/*------------------------------------------------------*/
int main()
{
    srand((unsigned)time(NULL));
    ofstream ofs, ofs_temperature;
    ifstream ifs;
    string str;
    vector<int>::iterator itr;

    vector<double> rad(N), Rx(N), Ry(N), Rz(N), Vx(N), Vy(N), Vz(N), q1(N), q2(N), q3(N), q4(N), Omega_x(N), Omega_y(N), Omega_z(N);
    vector<double> X_plus_x_t(N), X_plus_y_t(N), X_plus_z_t(N), X_plus_x_r(N), X_plus_y_r(N), X_plus_z_r(N);
    vector<double> Fx(N, 0.0), Fy(N, 0.0), Fz(N, 0.0), Tx(N, 0.0), Ty(N, 0.0), Tz(N, 0.0);
    vector<vector<int> > nn_list(N, vector<int>());
    vector<double> f_int(N * N * 3);

    vector<bool> isBond(N * N, false);
    vector<vector<int> > Bond(N, vector<int>());
    vector<double> s_bond(N * N * 3), t_bond(N * N * 3);
    double xi;

    int check = 0;

    int p, p1, p2;
    double Rx_old, Ry_old, Rz_old, Vx_h, Vy_h, Vz_h;
    double X_minus, V_plus, V_minus;

    double norm;
    double mp, Ip, gamma_t_p, gamma_r_p;
    double Omega_x_h, Omega_y_h, Omega_z_h, Omega_x_p, Omega_y_p, Omega_z_p, d_omega_x, d_omega_y, d_omega_z, q1_h, q2_h, q3_h, q4_h;

    int step;
    double t, t_diff, T_calc;

    vector<double> D(3), vec_tmp_p1(3), vec_tmp_p2(3);
    double r, s, t1_x, t1_y, t1_z, t2_x, t2_y, t2_z;

    /*paramater calculation*/
    double gamma_r = 10.0 * gamma_t / 3.0;
    double a1, a2, a3, a4, a5, a6, a7, a8, a9;
    double s1, s2, t1, t2;

    int ok;
    cout << "Initial Version\n";
    cout << "Check: N = " << N << "; L = " << L << "; phi = " << phi << "; p_bond = " << p_bond << "; k_angle = " << k_angle << "; k_tort = " << k_tort << "; kBT[K] = " << kBT << '\n';
    cout << "gamma_t = " << gamma_t << "; dt = " << dt << "; m = " << m << "; I = " << I << '\n';

    /*----------  Input Initial Condition  ----------*/
    sprintf(filename_input, "%s.dat", name_input_p);
    ifs.open(filename_input);
    while (getline(ifs, str))
    {
        replace(str.begin(), str.end(), ',', ' ');
        istringstream iss(str);
        iss >> p;
        iss >> rad[p];
        iss >> Rx[p] >> Ry[p] >> Rz[p];
        iss >> Vx[p] >> Vy[p] >> Vz[p];
        iss >> X_plus_x_t[p] >> X_plus_y_t[p] >> X_plus_z_t[p];
        iss >> q1[p] >> q2[p] >> q3[p] >> q4[p];
        iss >> Omega_x[p] >> Omega_y[p] >> Omega_z[p];
        iss >> X_plus_x_r[p] >> X_plus_y_r[p] >> X_plus_z_r[p];
    }
    ifs.close();

    sprintf(filename_input, "%s.dat", name_input_b);
    ifs.open(filename_input);
    while (getline(ifs, str))
    {
        replace(str.begin(), str.end(), ',', ' ');
        istringstream iss(str);
        iss >> p1 >> p2;
        iss >> s_bond[p1 * N * 3 + p2 * 3 + 0] >> s_bond[p1 * N * 3 + p2 * 3 + 1] >> s_bond[p1 * N * 3 + p2 * 3 + 2];
        iss >> t_bond[p1 * N * 3 + p2 * 3 + 0] >> t_bond[p1 * N * 3 + p2 * 3 + 1] >> t_bond[p1 * N * 3 + p2 * 3 + 2];
        iss >> s_bond[p2 * N * 3 + p1 * 3 + 0] >> s_bond[p2 * N * 3 + p1 * 3 + 1] >> s_bond[p2 * N * 3 + p1 * 3 + 2];
        iss >> t_bond[p2 * N * 3 + p1 * 3 + 0] >> t_bond[p2 * N * 3 + p1 * 3 + 1] >> t_bond[p2 * N * 3 + p1 * 3 + 2];

        isBond[p1 * N + p2] = true;
        isBond[p2 * N + p1] = true;
        Bond[p1].push_back(p2);
    }
    ifs.close();

    /*----------  PARAMETER and INITIAL CONDITION OUTPUT  -----------*/
    WRITE_parameter_init(rad, Rx, Ry, Rz, Vx, Vy, Vz, X_plus_x_t, X_plus_y_t, X_plus_z_t, q1, q2, q3, q4, Omega_x, Omega_y, Omega_z, X_plus_x_r, X_plus_y_r, X_plus_z_r);

    /*+++++++++++++++++++++++++++++++++*/
    /*----------  Main Loop  ----------*/
    /*+++++++++++++++++++++++++++++++++*/
    step = 0;
    sprintf(filename_temperature, "%s.dat", name_temperature);
    ofs_temperature.open(filename_temperature);
    ofs_temperature.close();

    while (1)
    {
        step += 1;
        K = U = 0.0;
        t = (double)step * dt;
        t_diff = t / tau_B;

        /*+++++  Calculation of Center Force  +++++*/
        U = calc_LJ_Force(rad, Rx, Ry, Rz, f_int, Fx, Fy, Fz, nn_list);

        /*+++++  Calculation of BOnd Force  +++++*/
        U += calc_Bond_Force(rad, Rx, Ry, Rz, q1, q2, q3, q4, Bond, s_bond, t_bond, Fx, Fy, Fz, Tx, Ty, Tz);

        /*+++++  Calculation of EOM and Initialization +++++*/
        for (p = 0; p < N; p++)
        {
            /*Translation Leap-frog*/
            mp = m * rad[p] * rad[p] * rad[p];
            gamma_t_p = gamma_t / (rad[p] * rad[p]);

            a1 = C(0.5 * gamma_t_p * dt);
            a2 = BoverC(0.5 * gamma_t_p * dt);
            a3 = one_minus_exp(gamma_t_p * dt);
            a4 = Bover_exp_minus(0.5 * gamma_t_p * dt);
            a5 = yDoverC(0.5 * gamma_t_p * dt);
            a6 = D_exp_minus_one(0.5 * gamma_t_p * dt);
            a7 = exp_exp(0.5 * gamma_t_p * dt);
            a8 = exp(-gamma_t_p * dt);

            s1 = C(0.5 * gamma_t_p * dt) * kBT / (mp * gamma_t_p * gamma_t_p);
            s2 = BoverC(0.5 * gamma_t_p * dt) * kBT / mp;
            t1 = one_minus_exp(gamma_t_p * dt) * kBT / mp;
            t2 = -1.0 * Bover_exp_minus(0.5 * gamma_t_p * dt) * kBT / (mp * gamma_t_p * gamma_t_p);

            Rx_old = Rx[p];
            Ry_old = Ry[p];
            Rz_old = Rz[p];

            V_minus = X_plus_x_t[p] * a5 * 2.0 / dt + Gauss(s2);
            V_plus = Gauss(t1);
            X_minus = V_plus * a6 / gamma_t_p + Gauss(t2);
            X_plus_x_t[p] = Gauss(s1);
            Vx[p] = Vx[p] * a8 + Fx[p] * a3 / (mp * gamma_t_p) + V_plus - V_minus * a8;
            Rx[p] += Vx[p] * a7 / gamma_t_p + X_plus_x_t[p] - X_minus;

            V_minus = X_plus_y_t[p] * a5 * 2.0 / dt + Gauss(s2);
            V_plus = Gauss(t1);
            X_minus = V_plus * a6 / gamma_t_p + Gauss(t2);
            X_plus_y_t[p] = Gauss(s1);
            Vy[p] = Vy[p] * a8 + Fy[p] * a3 / (mp * gamma_t_p) + V_plus - V_minus * a8;
            Ry[p] += Vy[p] * a7 / gamma_t_p + X_plus_y_t[p] - X_minus;

            V_minus = X_plus_z_t[p] * a5 * 2.0 / dt + Gauss(s2);
            V_plus = Gauss(t1);
            X_minus = V_plus * a6 / gamma_t_p + Gauss(t2);
            X_plus_z_t[p] = Gauss(s1);
            Vz[p] = Vz[p] * a8 + Fz[p] * a3 / (mp * gamma_t_p) + V_plus - V_minus * a8;
            Rz[p] += Vz[p] * a7 / gamma_t_p + X_plus_z_t[p] - X_minus;

            Vx_h = (Rx[p] - Rx_old) / dt;
            Vy_h = (Ry[p] - Ry_old) / dt;
            Vz_h = (Rz[p] - Rz_old) / dt;

            /* PBC */
            if (Rx[p] < 0)
                Rx[p] += L;
            if (Rx[p] >= L)
                Rx[p] -= L;
            if (Ry[p] < 0)
                Ry[p] += L;
            if (Ry[p] >= L)
                Ry[p] -= L;
            if (Rz[p] < 0)
                Rz[p] += L;
            if (Rz[p] >= L)
                Rz[p] -= L;

            if (Rx[p] < 0 || Rx[p] >= L || Ry[p] < 0 || Ry[p] >= L || Rz[p] < 0 || Rz[p] >= L)
            {
                cout << p << ", " << Rx_old << ", " << Ry_old << ", " << Rz_old << '\n';
                p1 = p;
                for(int i = 0; i < Bond[p1].size(); i++){
                    p2 = Bond[p1][i];
                    cout << p1 << "-> " << p2 << '\n';
                    vec_tmp_p1 = vec_p_to_r(s_bond[p1 * N * 3 + p2 * 3 + 0], s_bond[p1 * N * 3 + p2 * 3 + 1], s_bond[p1 * N * 3 + p2 * 3 + 2], q1[p1], q2[p1], q3[p1], q4[p1]);
                    cout << vec_tmp_p1[0] << ", " << vec_tmp_p1[1] << ", " << vec_tmp_p1[2] << '\n';
                    cout << p2 << "-> " << p1 << '\n';
                    vec_tmp_p1 = vec_p_to_r(s_bond[p2 * N * 3 + p1 * 3 + 0], s_bond[p2 * N * 3 + p1 * 3 + 1], s_bond[p2 * N * 3 + p1 * 3 + 2], q1[p2], q2[p2], q3[p2], q4[p2]);
                    cout << vec_tmp_p1[0] << ", " << vec_tmp_p1[1] << ", " << vec_tmp_p1[2] << '\n';
                }
                check = 1;
            }

            /* Rotation leap-frog */
            Ip = I * rad[p] * rad[p] * rad[p] * rad[p] * rad[p];
            gamma_r_p = gamma_r / (rad[p] * rad[p]);
            a1 = C(0.5 * gamma_r_p * dt);
            a2 = BoverC(0.5 * gamma_r_p * dt);
            a3 = one_minus_exp(gamma_r_p * dt);
            a4 = Bover_exp_minus(0.5 * gamma_r_p * dt);
            a5 = yDoverC(0.5 * gamma_r_p * dt);
            a6 = D_exp_minus_one(0.5 * gamma_r_p * dt);
            a7 = exp_exp(0.5 * gamma_r_p * dt);
            a8 = exp(-gamma_r_p * dt);
            a9 = one_minus_exp(0.5 * gamma_r_p * dt);

            s1 = a1 * kBT / (Ip * gamma_r_p * gamma_r_p);
            s2 = a2 * kBT / Ip;
            t1 = a3 * kBT / Ip;
            t2 = -a4 * kBT / (Ip * gamma_r_p * gamma_r_p);

            V_minus = X_plus_x_r[p] * a5 * 2.0 / dt + Gauss(s2);
            V_plus = Gauss(t1);
            X_minus = V_plus * a6 / gamma_r_p + Gauss(t2);
            X_plus_x_r[p] = Gauss(s1);
            Omega_x_h = Omega_x[p] * exp(-0.5 * gamma_r_p * dt) + Tx[p] * a9 / (Ip * gamma_r_p) + V_plus; // Omega_r (n)
            Omega_x[p] = Omega_x[p] * a8 + Tx[p] * a3 / (Ip * gamma_r_p) + V_plus - V_minus * a8;     // Omega_r (n + 0.5)
            d_omega_x = Omega_x[p] * a7 / gamma_r_p + X_plus_x_r[p] - X_minus;                            // d_omega_r

            V_minus = X_plus_y_r[p] * a5 * 2.0 / dt + Gauss(s2);
            V_plus = Gauss(t1);
            X_minus = V_plus * a6 / gamma_r_p + Gauss(t2);
            X_plus_y_r[p] = Gauss(s1);
            Omega_y_h = Omega_y[p] * exp(-0.5 * gamma_r_p * dt) + Ty[p] * a9 / (Ip * gamma_r_p) + V_plus;
            Omega_y[p] = Omega_y[p] * a8 + Ty[p] * a3 / (Ip * gamma_r_p) + V_plus - V_minus * a8;
            d_omega_y = Omega_y[p] * a7 / gamma_r_p + X_plus_y_r[p] - X_minus;

            V_minus = X_plus_z_r[p] * a5 * 2.0 / dt + Gauss(s2);
            V_plus = Gauss(t1);
            X_minus = V_plus * a6 / gamma_r_p + Gauss(t2);
            X_plus_z_r[p] = Gauss(s1);
            Omega_z_h = Omega_z[p] * exp(-0.5 * gamma_r_p * dt) + Tz[p] * a9 / (Ip * gamma_r_p) + V_plus;
            Omega_z[p] = Omega_z[p] * a8 + Tz[p] * a3 / (Ip * gamma_r_p) + V_plus - V_minus * a8;
            d_omega_z = Omega_z[p] * a7 / gamma_r_p + X_plus_z_r[p] - X_minus;

            conv_r_to_p(Omega_x_h, Omega_y_h, Omega_z_h, q1[p], q2[p], q3[p], q4[p]); // Omega_r (n) -> Omega_p (n)

            q1_h = q1[p] + 0.25 * dt * (-q3[p] * Omega_x_h - q4[p] * Omega_y_h + q2[p] * Omega_z_h);
            q2_h = q2[p] + 0.25 * dt * (q4[p] * Omega_x_h - q3[p] * Omega_y_h - q1[p] * Omega_z_h);
            q3_h = q3[p] + 0.25 * dt * (q1[p] * Omega_x_h + q2[p] * Omega_y_h + q4[p] * Omega_z_h);
            q4_h = q4[p] + 0.25 * dt * (-q2[p] * Omega_x_h + q1[p] * Omega_y_h - q3[p] * Omega_z_h); // q (n + 0.5)

            conv_r_to_p(d_omega_x, d_omega_y, d_omega_z, q1_h, q2_h, q3_h, q4_h); // d_omega_r -> d_omega_p

            q1[p] += 0.5 * (-q3_h * d_omega_x - q4_h * d_omega_y + q2_h * d_omega_z);
            q2[p] += 0.5 * (q4_h * d_omega_x - q3_h * d_omega_y - q1_h * d_omega_z);
            q3[p] += 0.5 * (q1_h * d_omega_x + q2_h * d_omega_y + q4_h * d_omega_z);
            q4[p] += 0.5 * (-q2_h * d_omega_x + q1_h * d_omega_y - q3_h * d_omega_z); // q (n + 1.0)

            norm = sqrt(q1[p] * q1[p] + q2[p] * q2[p] + q3[p] * q3[p] + q4[p] * q4[p]);

            q1[p] /= norm;
            q2[p] /= norm;
            q3[p] /= norm;
            q4[p] /= norm;

            K += 0.5 * mp * (Vx_h * Vx_h + Vy_h * Vy_h + Vz_h * Vz_h) + 0.5 * Ip * (Omega_x_h * Omega_x_h + Omega_y_h * Omega_y_h + Omega_z_h * Omega_z_h); // Kinetic energy

            Fx[p] = Fy[p] = Fz[p] = 0.0;
            Tx[p] = Ty[p] = Tz[p] = 0.0;
        }
        if (check == 1)
        {
            break;
        }

        Etot = K + U;
        T_calc = 1000 * K / (3 * (double)N * 8.3076);

        /* Bond Creation & Breaking */
        for (p1 = 0; p1 < N - 1; p1++)
        {
            for (int i = 0; i < nn_list[p1].size(); i++)
            {
                p2 = nn_list[p1][i];
                if (isBond[p1 * N + p2])
                {
                    vec_tmp_p1 = vec_p_to_r(s_bond[p1 * N * 3 + p2 * 3 + 0], s_bond[p1 * N * 3 + p2 * 3 + 1], s_bond[p1 * N * 3 + p2 * 3 + 2], q1[p1], q2[p1], q3[p1], q4[p1]);
                    vec_tmp_p2 = vec_p_to_r(s_bond[p2 * N * 3 + p1 * 3 + 0], s_bond[p2 * N * 3 + p1 * 3 + 1], s_bond[p2 * N * 3 + p1 * 3 + 2], q1[p2], q2[p2], q3[p2], q4[p2]);

                    D[0] = (Rx[p2] + rad[p2] * vec_tmp_p2[0]) - (Rx[p1] + rad[p1] * vec_tmp_p1[0]);
                    D[1] = (Ry[p2] + rad[p2] * vec_tmp_p2[1]) - (Ry[p1] + rad[p1] * vec_tmp_p1[1]);
                    D[2] = (Rz[p2] + rad[p2] * vec_tmp_p2[2]) - (Rz[p1] + rad[p1] * vec_tmp_p1[2]);
                    Dist_PBC(D);

                    s = sqrt(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);

                    if (s > s_max)
                    {
                        isBond[p1 * N + p2] = false;
                        isBond[p2 * N + p1] = false;

                        itr = Bond[p1].begin();
                        while (itr != Bond[p1].end())
                        {
                            if ((*itr) == p2)
                                itr = Bond[p1].erase(itr);
                            else
                                ++itr;
                        }
                    }
                }
                else
                {
                    D[0] = Rx[p2] - Rx[p1];
                    D[1] = Ry[p2] - Ry[p1];
                    D[2] = Rz[p2] - Rz[p1];
                    Dist_PBC(D);

                    r = sqrt(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);
                    s = r - (rad[p1] + rad[p2]);

                    if (s > s_equi - d_r && s < s_equi + d_r)
                    {
                        xi = (double)rand() / (double)RAND_MAX;
                        if (xi <= p_bond)
                        {
                            isBond[p1 * N + p2] = true;
                            isBond[p2 * N + p1] = true;
                            Bond[p1].push_back(p2);

                            D[0] /= r;
                            D[1] /= r;
                            D[2] /= r;
                            vec_tmp_p1 = vec_r_to_p(D[0], D[1], D[2], q1[p1], q2[p1], q3[p1], q4[p1]);
                            D[0] = -D[0];
                            D[1] = -D[1];
                            D[2] = -D[2];
                            vec_tmp_p2 = vec_r_to_p(D[0], D[1], D[2], q1[p2], q2[p2], q3[p2], q4[p2]);

                            s_bond[p1 * N * 3 + p2 * 3 + 0] = vec_tmp_p1[0];
                            s_bond[p1 * N * 3 + p2 * 3 + 1] = vec_tmp_p1[1];
                            s_bond[p1 * N * 3 + p2 * 3 + 2] = vec_tmp_p1[2]; // s_p1 vector at p1

                            s_bond[p2 * N * 3 + p1 * 3 + 0] = vec_tmp_p2[0];
                            s_bond[p2 * N * 3 + p1 * 3 + 1] = vec_tmp_p2[1];
                            s_bond[p2 * N * 3 + p1 * 3 + 2] = vec_tmp_p2[2]; // s_p2 vector at p2

                            t1_x = -s_bond[p1 * N * 3 + p2 * 3 + 1];
                            t1_y = s_bond[p1 * N * 3 + p2 * 3 + 0];
                            t1_z = 0.0;
                            norm = sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z);
                            if (norm != 0)
                            {
                                t1_x /= norm;
                                t1_y /= norm;
                                t1_z /= norm;
                            }
                            else
                            {
                                t1_x = 1.0;
                                t1_y = 0.0;
                                t1_z = 0.0;
                            } // t_p vector at p1
                            t_bond[p1 * N * 3 + p2 * 3 + 0] = t1_x;
                            t_bond[p1 * N * 3 + p2 * 3 + 1] = t1_y;
                            t_bond[p1 * N * 3 + p2 * 3 + 2] = t1_z;

                            vec_tmp_p1 = vec_p_to_r(t1_x, t1_y, t1_z, q1[p1], q2[p1], q3[p1], q4[p1]); // t_p1 vector at r
                            t2_x = -vec_tmp_p1[0];
                            t2_y = -vec_tmp_p1[1];
                            t2_z = -vec_tmp_p1[2];
                            norm = sqrt(t2_x * t2_x + t2_y * t2_y + t2_z * t2_z);
                            t2_x /= norm;
                            t2_y /= norm;
                            t2_z /= norm;                                                              // t_p2 vector at r
                            vec_tmp_p2 = vec_r_to_p(t2_x, t2_y, t2_z, q1[p2], q2[p2], q3[p2], q4[p2]); // t_p2 vector at p2
                            t_bond[p2 * N * 3 + p1 * 3 + 0] = vec_tmp_p2[0];
                            t_bond[p2 * N * 3 + p1 * 3 + 1] = vec_tmp_p2[1];
                            t_bond[p2 * N * 3 + p1 * 3 + 2] = vec_tmp_p2[2];
                        }
                    }
                }
            }
            nn_list[p1].clear();
        }
        nn_list[N - 1].clear();

        /*+++++  output  +++++*/
        if ((step % output_display) == 0)
        {
            cout << "step = " << step
                 << "; 0-th = " << Rx[0] << ",  " << Vx[0] << ",  " << q1[0] << ", " << Omega_x[0]
                 << "; Etot = " << Etot
                 << "; Temperature = " << T_calc
                 << '\n';
        }
        if ((step % write_file) == 0)
        {
            sprintf(filename, "%s_%05d.dat", name_p, (int)(step / write_file));
            ofs.open(filename);
            for (p = 0; p < N; p++)
            {
                ofs << p << ", "
                    << scientific << setprecision(15) 
                    << rad[p] << ", "
                    << Rx[p] << ", " << Ry[p] << ", " << Rz[p] << ", "
                    << Vx[p] << ", " << Vy[p] << ", " << Vz[p] << ", "
                    << X_plus_x_t[p] << ", " << X_plus_y_t[p] << ", " << X_plus_z_t[p] << ", "
                    << q1[p] << ", " << q2[p] << ", " << q3[p] << ", " << q4[p] << ", "
                    << Omega_x[p] << ", " << Omega_y[p] << ", " << Omega_z[p] << ", "
                    << X_plus_x_r[p] << ", " << X_plus_y_r[p] << ", " << X_plus_z_r[p] << ", "
                    << '\n';
            }
            ofs.close();

            sprintf(filename, "%s_%05d.dat", name_b, (int)(step / write_file));
            ofs.open(filename);
            for (p1 = 0; p1 < N - 1; p1++)
            {
                for (int i = 0; i < Bond[p1].size(); i++)
                {
                    p2 = Bond[p1][i];
                    ofs << p1 << ", " << p2 << ", "
                        << scientific << setprecision(15) 
                        << s_bond[p1 * N * 3 + p2 * 3 + 0] << ", " << s_bond[p1 * N * 3 + p2 * 3 + 1] << ", " << s_bond[p1 * N * 3 + p2 * 3 + 2] << ", "
                        << t_bond[p1 * N * 3 + p2 * 3 + 0] << ", " << t_bond[p1 * N * 3 + p2 * 3 + 1] << ", " << t_bond[p1 * N * 3 + p2 * 3 + 2] << ", "
                        << s_bond[p2 * N * 3 + p1 * 3 + 0] << ", " << s_bond[p2 * N * 3 + p1 * 3 + 1] << ", " << s_bond[p2 * N * 3 + p1 * 3 + 2] << ", "
                        << t_bond[p2 * N * 3 + p1 * 3 + 0] << ", " << t_bond[p2 * N * 3 + p1 * 3 + 1] << ", " << t_bond[p2 * N * 3 + p1 * 3 + 2] << ", \n";
                }
            }
            ofs.close();
        }

        if ((step % out_temperature) == 0)
        {
            int Nc = calc_Nc(Bond, N_bond);
            ofs_temperature.open(filename_temperature, ios::app);

            ofs_temperature << (int)(step / write_file) << ", " << t << ", " << t_diff << ", " << T_calc << ", " << Etot << ", " << N_bond << ", " << Nc << ",\n";

            ofs_temperature.close();
        }

    } // time step

    return 0;
}
