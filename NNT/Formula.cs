using System;

namespace NNT
{
    class Formula
    {
        public double Sigmoid(double x)
        {
            double y = 1 / (1 + Math.Exp(-x));

            return y;
        }

        public double InOutOld(double w1, double w2, int I1, int I2)
        {
            double Out = 0;

            Out = w1 * I1 + w2 * I2;

            return Out;
        }

        public double InOut(double[] w, int[] input)
        {
            double Out = 0;

            for (int i = 2; i < input.Length; i++)
            { 
                Out += input[i] * w[i];
            }

            return Out;
        }

        public double HideOut(double[] arr, double[] w)
        {
            double Out = 0;

            for (int i = 2; i < w.Length; i++)
            {
                Out += arr[i] * w[i];
            }

            return Out;
        }

        public double HideError()
        {
            double Out = 0;



            return Out;
        }
    }
}
