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

        public double InOut(double[] w, int[] input)
        {
            double Out = 0;

            for (int i = 0; i < input.Length; i++)
            { 
                Out += input[i] * w[i];
            }

            return Out;
        }

        public double HideOut(double[] arr, double[] w)
        {
            double Out = 0;

            for (int i = 0; i < w.Length; i++)
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
