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

        public double Out(double[] w, double[] input)
        {
            double Out = 0;

            for (int i = 2; i < input.Length; i++)
            { 
                Out += input[i] * w[i];
            }

            return Out;
        }
    }
}
