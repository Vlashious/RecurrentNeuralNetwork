using System;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;

namespace RecursiveNetwork
{
    class TargetMatrix
    {
        public int P { get; set; }
        public int L { get; set; }
        public Matrix<double> Matrix { get; set; }
        public TargetMatrix()
        {
            var jsonTxt = File.ReadAllText("matrix.conf");
            dynamic jsonObj = JsonConvert.DeserializeObject(jsonTxt);
            P = jsonObj["p"];
            L = jsonObj["L"];
            double[] seq = jsonObj["seq"].ToObject<double[]>();
            double[,] matrix = new double[L, P];
            var index = 0;
            for (int i = 0; i < L; i++)
            {
                for (int j = 0; j < P; j++, index++)
                {
                    matrix[i, j] = seq[index];
                }
                index -= P == 1 ? 1 : P - 1;
            }
            Matrix = Matrix<double>.Build.DenseOfArray(matrix);
            // Console.WriteLine(Matrix);
        }
    }
}
