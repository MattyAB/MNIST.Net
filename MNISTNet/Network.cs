using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MNISTNet
{
    internal class Network
    {
        // CONST DEFINITIONS
        public const int InputNeurons = 784; // 28 * 28
        public const int L1Neurons = 64;
        public const int L2Neurons = 32;
        public const int OutputNeurons = 10;

        public Matrix<double> W1;
        public Matrix<double> W2;
        public Matrix<double> W3;
        public Vector<double> B1;
        public Vector<double> B2;
        public Vector<double> B3;

        double[,] inputs;

        DataLoader dl;

        public Network(DataLoader dl)
        {
            this.dl = dl;

            inputs = new double[InputNeurons, dl.GetData("train").Count];

            W1 = Matrix<double>.Build.Random(L1Neurons, InputNeurons);
            W2 = Matrix<double>.Build.Random(L2Neurons, L1Neurons);
            W3 = Matrix<double>.Build.Random(OutputNeurons, L2Neurons);
            B1 = Vector<double>.Build.Random(L1Neurons);
            B2 = Vector<double>.Build.Random(L2Neurons);
            B3 = Vector<double>.Build.Random(OutputNeurons);

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
            Vector<double> X = ArrayToVector(input);

            Vector<double> Z2 = W1 * X;
            Vector<double> a2 = Sigmoid(Z2);
            Vector<double> Z3 = W2 * a2;
            Vector<double> a3 = Sigmoid(Z3);
            Vector<double> Z4 = W3 * a3;
            Vector<double> yHat = Sigmoid(Z4);

            double[] returner = yHat.ToArray();

            return returner;
        }

        public double[,] Forward(double[,] input)
        {
            Matrix<double> X = ArrayToMatrix(input);

            Matrix<double> Z2 = W1 * X;
            Matrix<double> a2 = Sigmoid(Z2);
            Matrix<double> Z3 = W2 * a2;
            Matrix<double> a3 = Sigmoid(Z3);
            Matrix<double> Z4 = W3 * a3;
            Matrix<double> yHat = Sigmoid(Z4);

            double[,] returner = yHat.ToArray();

            return returner;
        }

        public double Cost(List<Image> Trainables)
        {
            double cost = 0;
            
            /* Prep X and y data */

            double[,] TrainValues = new double[784, Trainables.Count];

            for (int i = 0; i < Trainables.Count; i++)
            {
                for (int j = 0; j < 784; j++)
                {
                    TrainValues[j, i] = Trainables[i].pixels[j];
                }
            }

            double[,] y = new double[10, Trainables.Count];

            for (int i = 0; i < Trainables.Count; i++)
            {
                y[Trainables[i].expected, i] = 1;
            }

            double[,] yHat = Forward(TrainValues);

            for (int i = 0; i < Trainables.Count; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    cost += Math.Pow(y[j, i] - yHat[j, i], 2);
                }
            }

            return cost;
        }

        public double Evaluate(List<Image> images)
        {
            /* Prep X and y data */

            double[,] TrainValues = new double[784, images.Count];

            for (int i = 0; i < images.Count; i++)
            {
                for (int j = 0; j < 784; j++)
                {
                    TrainValues[j, i] = images[i].pixels[j];
                }
            }

            double[,] yHat = Forward(TrainValues);

            int wins = 0;

            for (int i = 0; i < images.Count; i++)
            {
                int bestGuess = 0;

                for (int j = 0; j < 10; j++)
                {
                    if(yHat[bestGuess, i] > yHat[j, i])
                    {
                        bestGuess = j;
                    }
                }

                if(bestGuess == images[i].expected)
                {
                    wins++;
                }
            }

            double accuracy = (double)wins * 100 / (double)images.Count;
            return accuracy;
        }

        internal void TrainSlow(double n)
        {
            while (true)
            {
                List<Image> RowTrainer = dl.GetNextTrainBatch();

                double stdCost = Cost(RowTrainer);

                Console.WriteLine("Cost: " + stdCost + ", Accuracy: " + Evaluate(dl.GetData("test")) + "%");

                double[,] TrainValues = new double[784, RowTrainer.Count];

                for(int i = 0; i < RowTrainer.Count; i++)
                {
                    for(int j = 0; j < 784; j++)
                    {
                        TrainValues[j, i] = RowTrainer[i].pixels[j];
                    }
                }

                /* Get Pseudo-gradients of stuff */

                double[,] W1Edits = new double[W1.RowCount, W1.ColumnCount];

                int loops = 0;

                Parallel.For(0, W1.ColumnCount, i =>
                {
                    for (int j = 0; j < W1.RowCount; j++)
                    {
                        W1[j, i] += n;

                        double newCost = Cost(RowTrainer);

                        W1[j, i] -= n;

                        W1Edits[j, i] = stdCost - newCost;
                    }

                    loops++;

                    Console.WriteLine(loops);
                });

                double[,] W2Edits = new double[W2.RowCount, W2.ColumnCount];

                loops = 0;

                Parallel.For(0, W2.ColumnCount, i =>
                {
                    for (int j = 0; j < W2.RowCount; j++)
                    {
                        W2[j, i] += n;

                        double newCost = Cost(RowTrainer);

                        W2[j, i] -= n;

                        W2Edits[j, i] = stdCost - newCost;
                    }

                    loops++;

                    Console.WriteLine(loops);
                });

                double[,] W3Edits = new double[W3.RowCount, W3.ColumnCount];

                loops = 0;

                Parallel.For(0, W3.ColumnCount, i =>
                {
                    for (int j = 0; j < W3.RowCount; j++)
                    {
                        W3[j, i] += n;

                        double newCost = Cost(RowTrainer);

                        W3[j, i] -= n;

                        W3Edits[j, i] = stdCost - newCost;
                    }

                    loops++;

                    Console.WriteLine(loops);
                });

                /* Write new weights */

                for (int i = 0; i < W1.ColumnCount; i++)
                {
                    for (int j = 0; j < W1.RowCount; j++)
                    {
                        W1[j, i] -= n * W1Edits[j, i];
                    }
                }

                for (int i = 0; i < W2.ColumnCount; i++)
                {
                    for (int j = 0; j < W2.RowCount; j++)
                    {
                        W2[j, i] -= n * W2Edits[j, i];
                    }
                }

                for (int i = 0; i < W3.ColumnCount; i++)
                {
                    for (int j = 0; j < W3.RowCount; j++)
                    {
                        W3[j, i] -= n * W3Edits[j, i];
                    }
                }
            }
        }

        internal void Train(double n)
        {
            while (true)
            {
                List<Image> RowTrainer = dl.GetNextTrainBatch();

                double stdCost = Cost(RowTrainer);

                double learnRate;
                if (stdCost < 25)
                    learnRate = n;
                else
                    learnRate = n / 10;

                Console.WriteLine("Cost: " + stdCost + ", Accuracy: " + Evaluate(dl.GetData("test").GetRange(0, 100)) + "%");
                //Console.WriteLine("Cost: " + stdCost);

                double[,] TrainValues = new double[784, RowTrainer.Count];

                for (int i = 0; i < RowTrainer.Count; i++)
                {
                    for (int j = 0; j < 784; j++)
                    {
                        TrainValues[j, i] = RowTrainer[i].pixels[j];
                    }
                }

                /**
                List<Vector<double>> totalError = Backprop(RowTrainer[0]);
                for(int i = 1; i < RowTrainer.Count; i++)
                {
                    List<Vector<double>> errors = Backprop(RowTrainer[i]);
                    for (int j = 0; j < errors.Count; j++)
                        totalError[j] += errors[j];
                }
                
                for (int j = 0; j < totalError.Count; j++)
                    totalError[j] /= RowTrainer.Count;

                for (int i = 0; i < W1.RowCount; i++)
                {
                    for (int j = 0; j < W1.ColumnCount; j++)
                    {
                        W1[i,j] += n * totalError[0][]
                    }
                }
                **/
                
                for (int i = 1; i < RowTrainer.Count; i++)
                {
                    double[] XArr = new double[784];

                    for (int j = 0; j < 784; j++)
                    {
                        XArr[j] = RowTrainer[i].pixels[j];
                    }

                    Vector<double> X = ArrayToVector(XArr);

                    Vector<double> Z2 = W1 * X;
                    Vector<double> a2 = Sigmoid(Z2);
                    Vector<double> Z3 = W2 * a2;
                    Vector<double> a3 = Sigmoid(Z3);
                    Vector<double> Z4 = W3 * a3;
                    Vector<double> yHat = Sigmoid(Z4);

                    List<Vector<double>> errors = Backprop(RowTrainer[i]);

                    for (int j = 0; j < W1.RowCount; j++)
                    {
                        for (int k = 0; k < W1.ColumnCount; k++)
                        {
                            W1[j, k] += learnRate * errors[1][j] * X[k];
                        }
                    }

                    for (int j = 0; j < W2.RowCount; j++)
                    {
                        for (int k = 0; k < W2.ColumnCount; k++)
                        {
                            W2[j, k] += learnRate * errors[2][j] * a2[k];
                        }
                    }

                    for (int j = 0; j < W3.RowCount; j++)
                    {
                        for (int k = 0; k < W3.ColumnCount; k++)
                        {
                            W3[j, k] += learnRate * errors[3][j] * a3[k];
                        }
                    }
                }
            }
        }

        List<Vector<double>> Backprop(Image image)
        {
            /* Prep X and y data */

            double[] XArr = new double[784];

            for (int i = 0; i < 784; i++)
            {
                XArr[i] = image.pixels[i];
            }

            double[] yArr = new double[10];

            yArr[image.expected] = 1;
            
            Vector<double> X = ArrayToVector(XArr);

            Vector<double> Z2 = W1 * X;
            Vector<double> a2 = Sigmoid(Z2);
            Vector<double> Z3 = W2 * a2;
            Vector<double> a3 = Sigmoid(Z3);
            Vector<double> Z4 = W3 * a3;
            Vector<double> yHat = Sigmoid(Z4);

            double[] returner = yHat.ToArray();

            double[] yHatArr = Forward(XArr);
            
            Vector<double> y = ArrayToVector(yArr);

            Vector<double> yHatError = y - yHat;

            //Vector<double> Z4Error = SigmoidPrime(y) - SigmoidPrime(yHat);
            Vector<double> Z4Error = yHatError.PointwiseMultiply(TransferDerivative(Z4));

            Vector<double> a3Error = Vector<double>.Build.Random(L2Neurons);
            for (int i = 0; i < a3Error.Count; i++)
            {
                a3Error[i] = 0;
                for (int j = 0; j < Z4Error.Count; j++)
                {
                    a3Error[i] += W3[j, i] * Z4Error[j];
                }
            }

            Vector<double> Z3Error = a3Error.PointwiseMultiply(TransferDerivative(Z3));

            Vector<double> a2Error = Vector<double>.Build.Random(L1Neurons);
            for (int i = 0; i < a2Error.Count; i++)
            {
                a2Error[i] = 0;
                for (int j = 0; j < Z3Error.Count; j++)
                {
                    a2Error[i] += W2[j, i] * Z3Error[j];
                }
            }

            Vector<double> Z2Error = a2Error.PointwiseMultiply(TransferDerivative(Z2));

            Vector<double> a1Error = Vector<double>.Build.Random(InputNeurons);
            for (int i = 0; i < a1Error.Count; i++)
            {
                a1Error[i] = 0;
                for (int j = 0; j < Z2Error.Count; j++)
                {
                    a1Error[i] += W1[j, i] * Z2Error[j];
                }
            }

            Vector<double> Z1Error = a1Error.PointwiseMultiply(TransferDerivative(X));

            List<Vector<double>> output = new List<Vector<double>>();
            output.Add(Z1Error);
            output.Add(Z2Error);
            output.Add(Z3Error);
            output.Add(Z4Error);
            return output;
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

        private Vector<double> SigmoidPrime(Vector<double> x)
        {
            for (int i = 0; i < x.Count; i++)
            {
                x[i] = SpecialFunctions.Logit(x[i]);
            }

            return x;
        }

        private Vector<double> TransferDerivative(Vector<double> x)
        {
            for (int i = 0; i < x.Count; i++)
            {
                x[i] = x[i] * (1 - x[i]);
            }

            return x;
        }

        public Vector<double> ArrayToVector(double[] input)
        {
            return Vector<double>.Build.SparseOfArray(input);
        }

        public Matrix<double> ArrayToMatrix(double[,] input)
        {
            return Matrix<double>.Build.SparseOfArray(input);
        }
    }
}