using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using System;
using MathNet.Numerics;

namespace MNISTNet
{
    internal class NeuralNetwork
    {
        /**

        // CONST DEFINITIONS
        public const int InputNeurons = 784; // 28 * 28
        public const int L1Neurons = 64;
        public const int L2Neurons = 32;
        public const int OutputNeurons = 10;

        public Matrix<double> W1;
        public Matrix<double> W2;
        public Matrix<double> W3;
        public Matrix<double> B1;
        public Matrix<double> B2;
        public Matrix<double> B3;

        double[,] inputs;

        DataLoader dl;

        public NeuralNetwork(DataLoader dl)
        {
            this.dl = dl;

            inputs = new double[InputNeurons, dl.GetData("train").Count];
            //X = DenseMatrix.OfArray(new double[InputNeurons, dl.GetData("train").Count]);

            W1 = Matrix<double>.Build.Random(L1Neurons, InputNeurons);
            B1 = Matrix<double>.Build.Random(L1Neurons, InputNeurons);
            //Z2 = DenseMatrix.OfArray(new double[L1Neurons, dl.GetData("train").Count]);
            //a2 = DenseMatrix.OfArray(new double[L1Neurons, dl.GetData("train").Count]);

            W2 = Matrix<double>.Build.Random(L2Neurons, L1Neurons);
            B2 = Matrix<double>.Build.Random(L2Neurons, L1Neurons);
            //Z3 = DenseMatrix.OfArray(new double[L2Neurons, dl.GetData("train").Count]);
            //a3 = DenseMatrix.OfArray(new double[L2Neurons, dl.GetData("train").Count]);

            W3 = Matrix<double>.Build.Random(OutputNeurons, L2Neurons);
            B3 = Matrix<double>.Build.Random(OutputNeurons, L2Neurons);
            //Z4 = DenseMatrix.OfArray(new double[OutputNeurons, dl.GetData("train").Count]);
            //yHat = DenseMatrix.OfArray(new double[OutputNeurons, dl.GetData("train").Count]);

            for (int i = 0; i < dl.GetData("train").Count; i++)
            {
                Image rowData = dl.GetData("train", i);
                for (int j = 0; j < rowData.pixels.Length; j++)
                {
                    inputs[j, i] = rowData.pixels[j];
                }
            }
        }

        public double[] Forward(double[] input)
        {
            Vector<double> X = DenseVector.OfArray(input);

            Vector<double> Z2 = W1 * X;
            Vector<double> a2 = Sigmoid(Z2);
            Vector<double> Z3 = W2 * a2;
            Vector<double> a3 = Sigmoid(Z3);
            Vector<double> Z4 = W3 * a3;
            Vector<double> yHat = Sigmoid(Z4);

            double[] returner = yHat.ToArray();

            return returner;
        }

        /**
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
                    if (Math.Abs(yHat[j, i]) > yHat[actual, i])
                        actual = Math.Abs(j);
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

        public double Cost()
        {
            for(int i = 0; i < inputs.GetLength(0); i++)
            {
                double[] output = Forward()
            }

            double cost = 0;

            for(int i = 0; i < y.ColumnCount; i++)
            {
                cost += Math.Pow((y[0, i] - yHat[0, i]), 2);
            }

            return cost;
        }

        public void Train(double errorMargin)
        {
            
        }

        private Matrix<double> Sigmoid(Matrix<double> x)
        {
            for (int i = 0; i < x.ColumnCount; i++)
            {
                for (int j = 0; j < x.RowCount; j++)
                {
                    x[j, i] = SpecialFunctions.Logistic(x[j, i]);
                }
            }

            return x;
        }

        private Vector<double> Sigmoid(Vector<double> x)
        {
            for (int i = 0; i < x.Count; i++)
            {
                x[i] = SpecialFunctions.Logistic(x[i]);
            }

            return x;
        }

        private Matrix<double> SigmoidPrime(Matrix<double> x)
        {
            for (int i = 0; i < x.ColumnCount; i++)
            {
                for (int j = 0; j < x.RowCount; j++)
                {
                    x[j, i] = SpecialFunctions.Logit(x[j, i]);
                }
            }

            return x;
        }

        **/
    }
}