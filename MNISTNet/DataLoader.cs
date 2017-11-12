using System.IO;
using System;
using System.Collections.Generic;

namespace MNISTNet
{
    internal class DataLoader
    {
        List<Image> TrainData = new List<Image>();
        List<Image> TestData = new List<Image>();

        public DataLoader(string test, string train)
        {
            string[] trains = File.ReadAllLines(train);
            foreach (string line in trains)
            {
                string[] parts = line.Split(",");
                int[] ints = new int[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    ints[i] = Convert.ToInt32(parts[i]);
                }
                TrainData.Add(new Image(ints));
            }

            string[] tests = File.ReadAllLines(test);
            foreach (string line in trains)
            {
                string[] parts = line.Split(",");
                int[] ints = new int[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    ints[i] = Convert.ToInt32(parts[i]);
                }
                TestData.Add(new Image(ints));
            }
        }

        int batch = 0;
        const int batchSize = 20;

        public List<Image> GetNextTrainBatch()
        {
            int batchOffset = (batch * batchSize) % TrainData.Count;

            batch++;

            List<Image> output = new List<Image>();

            for (int i = batchOffset; i < batchOffset + batchSize; i++)
            {
                output.Add(TrainData[i]);
            }

            return output;
        }

        public Image GetData(string type, int location)
        {
            switch (type)
            {
                case "test":
                    return TestData[location];
                case "train":
                    return TrainData[location];
                default:
                    throw new Exception("Incorrect type passed to GetData()");
            }
        }

        public List<Image> GetData(string type)
        {
            switch (type)
            {
                case "test":
                    return TestData;
                case "train":
                    return TrainData;
                default:
                    throw new Exception("Incorrect type passed to GetData()");
            }
        }
    }

    class Image
    {
        public int expected;
        public double[] pixels = new double[28 * 28];

        public Image(int[] input)
        {
            if (input.Length != 785)
                throw new Exception("Incorrect number of inputs passed to image. Expected 785");

            List<int> data = new List<int>(input);

            // First number is expected digit from net.
            expected = data[0];
            data.RemoveAt(0);

            if (expected > 9 | expected < 0)
                throw new Exception("Bad digit passed to image.");

            for(int i = 0; i < data.Count; i++)
            {
                double tempdata = data[i];
                pixels[i] = Convert.ToDouble(tempdata / 255.0);
            }
        }
    }
}