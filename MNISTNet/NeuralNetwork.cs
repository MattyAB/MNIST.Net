using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using System;
using MathNet.Numerics;

namespace MNISTNet
{
    internal class NeuralNetwork
    {
        // CONST DEFINITIONS
        const int InputNeurons = 784; // 28 * 28
        const int L1Neurons = 64;
        const int L2Neurons = 32;
        const int OutputNeurons = 10;

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
            
            W1 = Matrix<double>.Build.Random(L1Neurons, InputNeurons);
            //Z2 = DenseMatrix.OfArray(new double[L1Neurons, dl.GetData("train").Count]);
            //a2 = DenseMatrix.OfArray(new double[L1Neurons, dl.GetData("train").Count]);
            
            W2 = Matrix<double>.Build.Random(L2Neurons, L1Neurons);
            //Z3 = DenseMatrix.OfArray(new double[L2Neurons, dl.GetData("train").Count]);
            //a3 = DenseMatrix.OfArray(new double[L2Neurons, dl.GetData("train").Count]);
            
            W3 = Matrix<double>.Build.Random(OutputNeurons, L2Neurons);
            //Z4 = DenseMatrix.OfArray(new double[OutputNeurons, dl.GetData("train").Count]);
            //yHat = DenseMatrix.OfArray(new double[OutputNeurons, dl.GetData("train").Count]);

            for (int i = 0; i < dl.GetData("train").Count; i++)
            {
                Image rowData = dl.GetData("train", i);
                for (int j = 0; j < rowData.pixels.Length; j++)
                {
                    X[j, i] = rowData.pixels[j];
                }
            }
        }

        public List<double> Forward()
        {
            Z2 = W1 * X;
            a2 = Sigmoid(Z2);
            Z3 = W2 * a2;
            a3 = Sigmoid(Z3);
            Z4 = W3 * a3;
            yHat = Sigmoid(Z4);

            List<double> returner = new List<double>();
            for (int i = 0; i < yHat.RowCount; i++)
            {
                returner.Add(yHat[i,0]);
            }

            return returner;
        }

        public double Accuracy()
        {
            double wins = 0;
            double total = 0;

            for(int i = 0; i < yHat.ColumnCount; i++)
            {
                total++;

                int expected = dl.GetData("train", i).expected;

                int actual = 0;
                for (int j = 0; j < yHat.RowCount; j++)
                {
                    if (yHat[j, i] > yHat[actual, i])
                        actual = j;
                }

                if (actual == expected)
                    wins++;
            }

            return wins / total;
        }

        public double Evaluate()
        {
            Forward();
            return Accuracy();
        }

        private Matrix<double> Sigmoid(Matrix<double> x)
        {
            for (int i = 0; i < x.ColumnCount; i++)
            {
                for (int j = 0; j < x.RowCount; j++)
                {
                    x[j, i] = SpecialFunctions.Logistic(x[j,i]);
                }
            }

            return x;
        }
    }
}