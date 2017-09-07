#ifndef __POLYNOMIAL_H__
#define __POLYNOMIAL_H__

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <assert.h>

class Polynomial
{
public:
    double& operator[](int power)
    {
        // Relies on the std::vector zero'ing the values
        if(power+1 > (int)m_coeffs.size()) {
            m_coeffs.resize(power+1);
        }

        return m_coeffs[power];
    }

    double operator[](int power) const
    {
        return m_coeffs[power];
    }

    double Eval(double x)
    {
        double ret = 0.0;
        double tmp_x = 1.0;

        for(size_t i=0; i < m_coeffs.size(); i++) {
            ret += m_coeffs[i]*tmp_x;
            tmp_x *= x;
        }

        return ret;
    }

    Polynomial operator*(const double val) const
    {
        Polynomial ans;

        for(size_t i=0; i < m_coeffs.size(); i++) {
            ans[i] = m_coeffs[i]*val;
        }

        return ans;
    }

    Polynomial& operator*=(const double val)
    {
        *this = *this * val;
        return *this;
    }

    Polynomial operator*(const Polynomial &rhs) const
    {
        Polynomial ans;

        for(size_t i=0; i < m_coeffs.size(); i++) {
            for(size_t j=0; j < rhs.m_coeffs.size(); j++) {
                double c = m_coeffs[i]*rhs.m_coeffs[j];
                ans[i+j] += c;
            }
        }

        return ans;
    }

    Polynomial& operator*=(const Polynomial &rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    Polynomial operator+(const Polynomial &rhs) const
    {
        Polynomial ans;

        if(m_coeffs.size() > rhs.m_coeffs.size()) {
            for(size_t i=0; i < m_coeffs.size(); i++) {
                ans[i] = m_coeffs[i] + rhs[i];
            }
        }
        else {
            for(size_t i=0; i < rhs.m_coeffs.size(); i++) {
                ans[i] = operator[](i) + rhs.m_coeffs[i];
            }
        }

        return ans;
    }

    Polynomial& operator+=(const Polynomial &rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    Polynomial operator-(const Polynomial &rhs) const
    {
        Polynomial ans;

        if(m_coeffs.size() > rhs.m_coeffs.size()) {
            for(size_t i=0; i < m_coeffs.size(); i++) {
                ans[i] = m_coeffs[i] - rhs[i];
            }
        }
        else {
            for(size_t i=0; i < rhs.m_coeffs.size(); i++) {
                ans[i] = operator[](i) - rhs.m_coeffs[i];
            }
        }

        return ans;
    }

    Polynomial& operator-=(const Polynomial &rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream &os, const Polynomial &p)
    {
        for(size_t i=0; i < p.m_coeffs.size(); i++) {
            os << i << ": " << p.m_coeffs[i] << std::endl;
        }

        return os;
    }

    std::vector <double> m_coeffs;
};

class PolyMatrix
{
public:
    PolyMatrix(int rows, int cols)
    {
        m_rows = rows;
        m_cols = cols;

        m_data.resize(rows);

        for(int i=0; i < rows; i++) {
            m_data[i].resize(cols);
        }
    }

    Polynomial& operator()(int row, int col)
    {
        return m_data[row][col];
    }

    void Eval(double x, double *ret)
    {
        for(int i=0; i < m_rows; i++) {
            for(int j=0; j < m_cols; j++) {
                ret[i*m_cols + j] = m_data[i][j].Eval(x);
            }
        }
    }

    friend std::ostream& operator<<(std::ostream &os, const PolyMatrix &p)
    {
        for(size_t i=0; i < p.m_data.size(); i++) {
            for(size_t j=0; j < p.m_data[i].size(); j++) {
                os << std::fixed << std::setprecision(4) << p.m_data[i][j][0] << " ";
            }

            os << std::endl;
        }

        return os;
    }

    int m_rows, m_cols;
    std::vector < std::vector<Polynomial> > m_data;
};

#endif
