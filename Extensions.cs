using System;
using MathNet.Numerics.LinearAlgebra;

namespace RecursiveNetwork
{
    static class Extensions
    {
        public static Matrix<double> Activate(this Matrix<double> matrix)
        {
            matrix = matrix.Map(f => Function(f));
            return matrix;
        }

        public static Matrix<double> ElementMult(this Matrix<double> matrix, Matrix<double> other)
        {
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] *= other[i, j];
                }
            }
            return matrix;
        }

        public static Matrix<double> ActivateDer(this Matrix<double> matrix)
        {
            matrix = matrix.Map(f => FunctionDer(f));
            return matrix;
        }

        public static double Function(double x) => Math.Log(x + Math.Sqrt(x * x + 1));
        public static double FunctionDer(double x) => 1f / (Math.Sqrt(x * x + 1));

        // public static double Function(double x) => 1f / (1 + Math.Exp(-x));
        // public static double FunctionDer(double x) => Function(x) * (1 - Function(x));

        // public static double Function(double x) => x;
        // public static double FunctionDer(double x) => 1;
    }
}