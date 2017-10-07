using System;

namespace MNISTNet
{
    class Program
    {
        const string test = @"D:\Documents\GitHub\MNISTNet\test.csv";
        const string train = @"D:\Documents\GitHub\MNISTNet\train.csv";

        static void Main(string[] args)
        {
            DataLoader dl = new DataLoader(test, train);

            Console.WriteLine("Loaded all data.");

            NeuralNetwork nn = new NeuralNetwork(dl);

            Console.WriteLine("Hello World!");
            Console.ReadLine();
        }
    }
}
