using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;

namespace MNISTNet
{
    internal class NeuralNetwork
    {
        // CONST DEFINITIONS
        const int InputNeurons = 784; // 28 * 28
        const int L1Neurons = 64;
        const int L2Neurons = 32;
        const int OutputNeurons = 32;

        Matrix<double> X;
        Matrix<double> W1;
        Matrix<double> Z2;
        Matrix<double> a2;
        Matrix<double> W2;
        Matrix<double> Z3;
        Matrix<double> a3;
        Matrix<double> W3;
        Matrix<double> Z4;
        Matrix<double> yHat;
        DataLoader dl;

        public NeuralNetwork(DataLoader dl)
        {
            this.dl = dl;

            X = DenseMatrix.OfArray(new double[InputNeurons, dl.GetData("train").Count]);

            W1 = DenseMatrix.OfArray(new double[L1Neurons, InputNeurons]);
            Z2 = DenseMatrix.OfArray(new double[L1Neurons, dl.GetData("train").Count]);
            a2 = DenseMatrix.OfArray(new double[L1Neurons, dl.GetData("train").Count]);

            W2 = DenseMatrix.OfArray(new double[L2Neurons, L1Neurons]);
            Z3 = DenseMatrix.OfArray(new double[L2Neurons, dl.GetData("train").Count]);
            a3 = DenseMatrix.OfArray(new double[L2Neurons, dl.GetData("train").Count]);

            W3 = DenseMatrix.OfArray(new double[OutputNeurons, L2Neurons]);
            Z4 = DenseMatrix.OfArray(new double[OutputNeurons, dl.GetData("train").Count]);
            yHat = DenseMatrix.OfArray(new double[OutputNeurons, dl.GetData("train").Count]);

            for (int i = 0; i < W1.ColumnCount; i++)
            {
                for (int j = 0; j < W1.RowCount; j++)
                {
                    W1[i, j] = 1;
                }
            }

            for (int i = 0; i < dl.GetData("train").Count; i++)
            {
                Image rowData = dl.GetData("train", i);
                for (int j = 0; j < rowData.pixels.Length; j++)
                {
                    X[i, j] = rowData.pixels[j];
                }
            }
        }


    }
}