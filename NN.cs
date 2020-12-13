using System;
using System.IO;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;

namespace RecursiveNetwork
{
    class NN
    {
        public double Rate { get; set; }
        public double FadeRate { get; set; }
        public double Error { get; set; }
        public int Epochs { get; set; }
        public int Z { get; set; }
        private Matrix<double> Matrix { get; set; }
        private Matrix<double> W { get; set; }
        private Matrix<double> V { get; set; }
        private Matrix<double> JordanCtx { get; set; }
        private Matrix<double> ElmanCtx { get; set; }
        public NN(Matrix<double> matrix)
        {
            var json = File.ReadAllText("nn.conf");
            dynamic jsonObj = JsonConvert.DeserializeObject(json);
            Rate = jsonObj["rate"];
            FadeRate = jsonObj["fade"];
            Error = jsonObj["mse"];
            Epochs = jsonObj["epochs"];
            Z = jsonObj["z"];

            Matrix = matrix;
            W = Matrix<double>.Build.Random(Matrix.ColumnCount + Matrix.RowCount, Matrix.RowCount, new ContinuousUniform(-1, 1));
            V = Matrix<double>.Build.Random(Matrix.RowCount, 1, new ContinuousUniform(-1, 1));

            JordanCtx = Matrix<double>.Build.Dense(1, 1);
            ElmanCtx = Matrix<double>.Build.Dense(1, Matrix.RowCount);
        }
        public void Train()
        {
            var error = double.MaxValue;
            while (error > Error && Epochs > 0)
            {
                error = 0;
                Epochs--;
                ClearContext();
                foreach (var vec in Matrix.ToRowArrays())
                {
                    ProcessVectorTrain(vec);
                }
                ClearContext();
                foreach (var vec in Matrix.ToRowArrays())
                {
                    error += ProcessVectorError(vec);
                }
                Console.WriteLine($"Error is: {error}.");
                Rate -= FadeRate;
            }
            Predict();
        }

        private void Predict()
        {
            ClearContext();
            foreach (var vec in Matrix.ToRowArrays())
            {
                ProcessVector(vec, out _);
            }
            var lastVec = Matrix.ToRowArrays().Last();
            Array.Copy(lastVec, 1, lastVec, 0, lastVec.Length - 1);
            for (int i = 0; i < Z; i++)
            {
                ProcessVector(lastVec, out double outVal);
                outVal = Math.Round(outVal);
                Console.WriteLine($"Next num is: {outVal}");

                Array.Copy(lastVec, 1, lastVec, 0, lastVec.Length - 1);
                lastVec[lastVec.Length - 2] = outVal;
            }
            Console.WriteLine(Matrix);
        }

        private void ClearContext()
        {
            JordanCtx = Matrix<double>.Build.Dense(1, 1);
            ElmanCtx = Matrix<double>.Build.Dense(1, Matrix.RowCount);
        }

        private double ProcessVectorError(double[] vec)
        {
            var inputs = Matrix<double>.Build.Dense(1, vec.Length - 1, vec.Take(vec.Length - 1).ToArray());
            var target = vec.Last();

            inputs = inputs.Append(ElmanCtx).Append(JordanCtx);
            var hiddenNet = inputs * W;
            var hiddenOut = hiddenNet.Activate();
            ElmanCtx = hiddenOut;
            var netVal = hiddenOut * V;
            var outVal = netVal.Activate();
            JordanCtx = outVal;

            return 1f / 2 * Math.Pow(target - outVal[0, 0], 2);
        }

        private void ProcessVectorTrain(double[] vec)
        {
            var inputs = Matrix<double>.Build.Dense(1, vec.Length - 1, vec.Take(vec.Length - 1).ToArray());
            var target = vec.Last();

            inputs = inputs.Append(ElmanCtx).Append(JordanCtx);
            var hiddenNet = inputs * W;
            var hiddenOut = hiddenNet.Activate();
            ElmanCtx = hiddenOut;
            var netVal = hiddenOut * V;
            var outVal = netVal.Activate();
            JordanCtx = outVal;

            var dW = GetDW(target, outVal[0, 0], netVal[0, 0], hiddenNet, inputs);
            W -= Rate * dW;

            var dV = GetDV(target, outVal[0, 0], netVal[0, 0], hiddenOut);
            V -= Rate * dV;
        }

        private void ProcessVector(double[] vec, out double result)
        {
            var inputs = Matrix<double>.Build.Dense(1, vec.Length - 1, vec.Take(vec.Length - 1).ToArray());

            inputs = inputs.Append(ElmanCtx).Append(JordanCtx);
            var hiddenNet = inputs * W;
            var hiddenOut = hiddenNet.Activate();
            ElmanCtx = hiddenOut;
            var netVal = hiddenOut * V;
            var outVal = netVal.Activate();
            JordanCtx = outVal;
            result = outVal[0, 0];
        }

        private Matrix<double> GetDV(in double target, in double outVal, in double netVal, in Matrix<double> hiddenOut)
        {
            var dV = Matrix<double>.Build.Dense(V.RowCount, V.ColumnCount);
            for (int i = 0; i < V.RowCount; i++)
            {
                for (int j = 0; j < V.ColumnCount; j++)
                {
                    var dE_dOut = -(target - outVal);
                    var dOut_dNet = Extensions.FunctionDer(netVal);
                    var dNet_dVij = hiddenOut[j, i];
                    var dE_dVij = dE_dOut * dOut_dNet * dNet_dVij;
                    dV[i, j] = dE_dVij;
                }
            }

            return dV;
        }

        private Matrix<double> GetDW(in double target, in double outVal, in double netVal, in Matrix<double> hiddenNet, in Matrix<double> input)
        {
            var dW = Matrix<double>.Build.Dense(W.RowCount, W.ColumnCount);
            for (int i = 0; i < W.RowCount; i++)
            {
                for (int j = 0; j < W.ColumnCount; j++)
                {
                    var dE_dOut = -(target - outVal);
                    var dOut_dNet = Extensions.FunctionDer(netVal);
                    var dNet_dOutW = V[j, 0];
                    var dOutW_dNetW = Extensions.FunctionDer(hiddenNet[0, j]);
                    var dNetW_dWij = input[0, i];
                    var dWij = dE_dOut * dOut_dNet * dNet_dOutW * dOutW_dNetW * dNetW_dWij;
                    dW[i, j] = dWij;
                }
            }
            return dW;
        }
    }
}