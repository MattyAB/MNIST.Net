using MathNet.Numerics.LinearAlgebra;
using System;

namespace MNISTNet
{
    internal class GeneticTrainer
    {
        /**

        NeuralNetwork CurrentNetwork;

        public GeneticTrainer(NeuralNetwork net)
        {
            CurrentNetwork = net;
        }

        internal void Train(int populationSize)
        {
            NeuralNetwork[] nets = new NeuralNetwork[populationSize];
            double[] results = new double[populationSize];

            for (int i = 0; i < populationSize; i++)
            {
                if(i % (populationSize / 100) == 0)
                {
                    Console.Clear();
                    double ii = i;
                    double poppop = populationSize;
                    Console.WriteLine((ii / poppop) * 100 + "%");
                }

                nets[i] = CurrentNetwork;

                nets[i].W1 = Matrix<double>.Build.Random(NeuralNetwork.L1Neurons, NeuralNetwork.InputNeurons);
                nets[i].W2 = Matrix<double>.Build.Random(NeuralNetwork.L2Neurons, NeuralNetwork.L1Neurons);
                nets[i].W3 = Matrix<double>.Build.Random(NeuralNetwork.OutputNeurons, NeuralNetwork.L2Neurons);

                results[i] = nets[i].Evaluate();
            }

            int best = 0;
            for(int i = 0; i < results.Length; i++)
            {
                if (results[i] > results[best])
                    best = i;
            }

            Console.WriteLine("Best network: " + results[best]);

            CurrentNetwork = nets[best];
        }
    **/
    }
}