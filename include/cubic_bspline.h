#pragma once

template <typename T>
inline T cubic_bspline(T r, T fac)
{
    T ret = 0.0;
    if (r <= 1 && r >= 0){
        ret = (1 - 1.5 * r * r *(1 - 0.5 *r)) * fac;
    }
    else if (r > 1 && r <= 2){
        ret = (2-r)*(2-r)*(2-r) * fac /4;
    }
    return ret;
}

template <typename T>
inline T cubic_bspline_derivative(T r, T fac)
{
    T ret = 0.0;

    if (r <= 1 && r > 0) {
        ret = -fac * (3*r - 9 * r * r/4);
    }
    else if (r >= 1 && r <= 2){
        ret = -fac * 0.75 * (2 - r) * (2 - r);
    }
    return ret;
}

template <typename T>
inline T cubic_bspline_hessian(T r, T fac)
{
    T ret = 0.0;

    if (r <= 1 && r > 0) {
        ret = -fac * (3 - 9 * r / 2);
    }
    else if (r >= 1 && r <= 2){
        ret = fac * 1.5 * (2 - r);
    }
    return ret;
}